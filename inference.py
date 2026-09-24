import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from motif_plus import QFormerConfig, MoTIFPlusForConditionalGeneration, disable_torch_init
from data.instruct_dataset import strip_media_tags, build_prompt


QWEN3VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
LLM_PATH = "./output/stage1/final_model"
QFORMER_CKPT = "./output/stage2/qformer.pt"
LORA_PATH = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3vl_path", type=str, default=QWEN3VL_PATH)
    parser.add_argument("--llm_path", type=str, default=LLM_PATH)
    parser.add_argument("--qformer_ckpt", type=str, default=QFORMER_CKPT)
    parser.add_argument("--lora_path", type=str, default=LORA_PATH)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--num_query", type=int, default=32)
    parser.add_argument("--qformer_hidden", type=int, default=768)
    parser.add_argument("--qformer_layers", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    disable_torch_init()

    print("Loading model ...")
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

    prompt = build_prompt(args.question)
    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    proc = processor(videos=[args.video], return_tensors="pt")
    pixel_values = (
        proc["pixel_values_videos"]
        if "pixel_values_videos" in proc
        else proc["pixel_values"]
    ).to(device)

    print("Generating ...")
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(generated)


if __name__ == "__main__":
    main()
