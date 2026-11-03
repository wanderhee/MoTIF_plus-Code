import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score import parse_prediction


def load_gold(path, split, subset):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    videos = [v for v in data["videos"] if v.get("scorable")]
    if split != "all":
        videos = [v for v in videos if v.get("split") == split]
    if subset:
        videos = [v for v in videos if v.get("subset") == subset]
    return videos


def load_model(args):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen3VLForConditionalGeneration,
    )
    from motif_plus import QFormerConfig, MoTIFPlusForConditionalGeneration, disable_torch_init

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

    if args.qformer_ckpt:
        ckpt = torch.load(args.qformer_ckpt, map_location="cpu")
        model.qformer.load_state_dict(ckpt["qformer"])
        model.llm_proj.load_state_dict(ckpt["llm_proj"])
        model.frame_embeddings.data.copy_(ckpt["frame_embeddings"])
    if args.lora_path:
        from peft import PeftModel

        model.llm = PeftModel.from_pretrained(model.llm, args.lora_path)
        model.llm = model.llm.merge_and_unload()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    return model, processor, tokenizer, device


def build_inputs(processor, tokenizer, video_path, prompt, device):
    full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    proc = processor(videos=[str(video_path)], return_tensors="pt")
    pixel_values = proc["pixel_values_videos"] if "pixel_values_videos" in proc else proc["pixel_values"]
    return input_ids, pixel_values.to(device)


def find_events(video_path, events_dir):
    directory = Path(events_dir)
    candidates = [
        directory / f"{video_path.stem}_events.json",
        directory / f"{video_path.parent.name}_{video_path.stem}_events.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def build_epf_prompt(video_path, base_prompt, args, prompt_manager, builder):
    from pipeline import adapt, load_events, read_media_meta, run_perception

    events_path = find_events(video_path, args.events_dir)
    if events_path is None:
        events_path = run_perception(
            video_path,
            args.weights,
            Path(args.perception_output),
        )
    events = load_events(events_path)
    event_data = adapt(events)
    media_meta = read_media_meta(video_path)
    package = builder.build(event_data, "视频", media_meta)
    prefix = package["P_final"].split("【P_output")[0].rstrip()
    bridge = "请结合上方的 YOLO 检测结果、初步事件假设和 EPF 证据 token，观看原始视频，并按要求输出结构化结果。"
    return prefix + "\n\n" + bridge + "\n\n" + base_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(ROOT / "evaluation" / "structured.json"))
    parser.add_argument("--prompt", default=str(ROOT / "evaluation" / "prompt_template.txt"))
    parser.add_argument("--video_root", default=str(ROOT / "datasets" / "videos"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--subset", default="shandong_4qa")
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "predictions.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--use_epf", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--events_dir", default=str(ROOT / "evaluation" / "events"))
    parser.add_argument("--perception_output", default=str(ROOT / "evaluation" / "perception"))
    parser.add_argument("--weights", default=str(ROOT / "perception" / "weights" / "yolov10n-shangao-v6.pt"))
    parser.add_argument("--epf_config", default=str(ROOT / "epf" / "prompts_config_0121.json"))

    parser.add_argument("--qwen3vl_path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--llm_path", default=str(ROOT / "output" / "stage1" / "final_model"))
    parser.add_argument("--qformer_ckpt", default=str(ROOT / "output" / "stage2" / "qformer.pt"))
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--num_query", type=int, default=32)
    parser.add_argument("--qformer_hidden", type=int, default=768)
    parser.add_argument("--qformer_layers", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    videos = load_gold(args.gold, args.split, args.subset)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit("no videos")

    base_prompt = Path(args.prompt).read_text(encoding="utf-8")
    video_root = Path(args.video_root)
    Path(args.events_dir).mkdir(parents=True, exist_ok=True)
    Path(args.perception_output).mkdir(parents=True, exist_ok=True)

    prompt_manager = None
    builder = None
    if args.use_epf:
        from epf import EPFBuilder, PromptManager

        prompt_manager = PromptManager(args.epf_config)
        builder = EPFBuilder(prompt_manager)

    import torch

    model, processor, tokenizer, device = load_model(args)

    predictions = []
    errors = 0
    for item in videos:
        video_path = video_root / item["video"]
        if not video_path.exists():
            errors += 1
            predictions.append({
                "video": item["video"],
                "prediction": None,
                "error": "video not found",
            })
            print(f"missing: {video_path}")
            continue
        try:
            if args.use_epf:
                prompt_text = build_epf_prompt(
                    video_path, base_prompt, args, prompt_manager, builder
                )
            else:
                prompt_text = base_prompt
            input_ids, pixel_values = build_inputs(
                processor, tokenizer, video_path, prompt_text, device
            )
            with torch.no_grad():
                output_ids = model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            raw = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            parsed = parse_prediction(raw)
            predictions.append({
                "video": item["video"],
                "prediction": parsed if parsed is not None else {},
                "raw_output": raw,
            })
        except Exception as exc:
            errors += 1
            predictions.append({
                "video": item["video"],
                "prediction": None,
                "error": str(exc),
            })
            print(f"error: {item['video']}: {exc}")

    payload = {
        "meta": {
            "gold": args.gold,
            "split": args.split,
            "subset": args.subset,
            "n_videos": len(videos),
            "n_errors": errors,
            "epf": args.use_epf,
            "events_dir": args.events_dir,
            "weights": args.weights,
            "qwen3vl_path": args.qwen3vl_path,
            "llm_path": args.llm_path,
            "qformer_ckpt": args.qformer_ckpt,
            "lora_path": args.lora_path,
        },
        "predictions": predictions,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(videos)} videos, {errors} errors")
    print(f"saved: {out}")

    if args.score:
        report = ROOT / "evaluation" / f"report_{args.split}"
        cmd = [
            sys.executable,
            str(ROOT / "evaluation" / "score.py"),
            "--pred", str(out),
            "--split", args.split,
            "--subset", args.subset,
            "--out", str(report),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
