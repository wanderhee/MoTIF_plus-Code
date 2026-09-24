import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
PERCEPTION_DIR = ROOT / "perception"

EVENT_TYPE_MAP = {
    "breakdown": 41,
    "people": 2,
    "parked": 4,
    "jam": 1,
}
EVENT_PRIORITY = ("breakdown", "people", "parked", "jam")

QWEN3VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
LLM_PATH = "./output/stage1/final_model"
QFORMER_CKPT = "./output/stage2/qformer.pt"
LORA_PATH = None


def run_perception(video: Path, weights: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "EventDet.py",
        "--weights", weights,
        "--source", str(video),
        "--output", str(output_dir),
    ]
    subprocess.run(cmd, cwd=str(PERCEPTION_DIR), check=True)
    return output_dir / (video.stem + "_events.json")


def load_events(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def aggregate_event_type(result: list) -> int:
    triggered = set()
    for frame in result:
        event = frame.get("event") or {}
        for name in EVENT_TYPE_MAP:
            if event.get(name):
                triggered.add(name)
    for name in EVENT_PRIORITY:
        if name in triggered:
            return EVENT_TYPE_MAP[name]
    return 0


def pick_frame(result: list, event_type: int) -> dict:
    for name, et in EVENT_TYPE_MAP.items():
        if et == event_type:
            for frame in result:
                if (frame.get("event") or {}).get(name):
                    return frame
    return max(result, key=lambda f: len(f.get("objects") or []), default={})


def convert_object(obj: dict) -> dict:
    bbox = obj.get("bbox") or {}
    x1, y1 = bbox.get("x_min"), bbox.get("y_min")
    x2, y2 = bbox.get("x_max"), bbox.get("y_max")
    det = {
        "class": obj.get("class_name", "unknown"),
        "confidence": obj.get("confidence", 0.0),
        "track_id": obj.get("track_id"),
        "speed": obj.get("speed"),
    }
    if None not in (x1, y1, x2, y2):
        det["bbox"] = [x1, y1, x2, y2]
    return det


def adapt(events: dict) -> dict:
    result = events.get("result") or []
    event_type = aggregate_event_type(result)
    frame = pick_frame(result, event_type)
    objects = [convert_object(o) for o in (frame.get("objects") or [])]
    return {"eventType": event_type, "objects": objects}


def read_media_meta(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"width": width, "height": height}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--weights", type=str, default="weights/yolov10n-shangao-v6.pt")
    parser.add_argument("--config", type=str, default=str(ROOT / "epf" / "prompts_config_0121.json"))
    parser.add_argument("--output", type=str, default="output/perception")
    parser.add_argument("--events_json", type=str, default=None,
                        help="Skip perception and read an existing *_events.json.")

    parser.add_argument("--qwen3vl_path", type=str, default=QWEN3VL_PATH)
    parser.add_argument("--llm_path", type=str, default=LLM_PATH)
    parser.add_argument("--qformer_ckpt", type=str, default=QFORMER_CKPT)
    parser.add_argument("--lora_path", type=str, default=LORA_PATH)
    parser.add_argument("--num_query", type=int, default=32)
    parser.add_argument("--qformer_hidden", type=int, default=768)
    parser.add_argument("--qformer_layers", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def run_mllm(video: Path, prompt_text: str, args, output_dir: Path) -> str:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen3VLForConditionalGeneration,
    )
    from motif_plus import (
        QFormerConfig,
        MoTIFPlusForConditionalGeneration,
        disable_torch_init,
    )

    disable_torch_init()

    qwen3vl = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen3vl_path, torch_dtype=torch.bfloat16
    )
    visual = qwen3vl.visual
    processor = AutoProcessor.from_pretrained(args.qwen3vl_path)

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)

    d_visual = getattr(visual.config, "hidden_size", None) or visual.config.embed_dim
    qformer_config = QFormerConfig(
        d_visual=d_visual,
        num_query=args.num_query,
        hidden_size=args.qformer_hidden,
        num_layers=args.qformer_layers,
    )
    model = MoTIFPlusForConditionalGeneration(visual, llm, qformer_config)

    ckpt = torch.load(args.qformer_ckpt, map_location="cpu")
    model.qformer.load_state_dict(ckpt["qformer"])
    model.llm_proj.load_state_dict(ckpt["llm_proj"])
    model.frame_embeddings.data.copy_(ckpt["frame_embeddings"])

    if args.lora_path is not None:
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, args.lora_path)
        model.llm = model.llm.merge_and_unload()

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    proc = processor(videos=[str(video)], return_tensors="pt")
    pixel_values = (
        proc["pixel_values_videos"]
        if "pixel_values_videos" in proc
        else proc["pixel_values"]
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)


def main():
    args = parse_args()
    video = Path(args.video)

    if args.events_json:
        print(f"Loading events from {args.events_json} ...")
        events = load_events(Path(args.events_json))
    else:
        print("Running YOLO perception ...")
        events = load_events(run_perception(video, args.weights, Path(args.output)))

    event_data = adapt(events)
    media_meta = read_media_meta(video)

    from epf import EPFBuilder, PromptManager

    prompt_manager = PromptManager(args.config)
    builder = EPFBuilder(prompt_manager)
    package = builder.build(event_data, "视频", media_meta)

    print("=== EPF structured evidence (D / E / Z_EPF) ===")
    print(json.dumps(
        {"D": package["D"], "E": package["E"], "Z_EPF": package["Z_EPF"]},
        ensure_ascii=False, indent=2,
    ))
    print("\n=== P_final (prompt for the MLLM) ===")
    print(package["P_final"])

    out = Path(args.output) / (video.stem + "_epf.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEPF package saved to: {out}")

    if args.qformer_ckpt:
        print("\nRunning MLLM secondary verification ...")
        report = run_mllm(video, package["P_final"], args, Path(args.output))
        print("\n=== MLLM verification report ===")
        print(report)
        report_path = Path(args.output) / (video.stem + "_report.txt")
        report_path.write_text(report, encoding="utf-8")
        print(f"\nReport saved to: {report_path}")
    else:
        print("\n(--qformer_ckpt not provided; skipping MLLM verification, EPF prompt only.)")


if __name__ == "__main__":
    main()
