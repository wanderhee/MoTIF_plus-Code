import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motif_plus import QFormerConfig, MoTIFPlusForConditionalGeneration, disable_torch_init


QWEN3VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
LLM_PATH = "./output/stage1/final_model"
ALIGN_JSON = "./datasets/videomodified.json"
DATA_FOLDER = "./datasets"
OUTPUT_DIR = "./output/stage2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3vl_path", type=str, default=QWEN3VL_PATH)
    parser.add_argument("--llm_path", type=str, default=LLM_PATH)
    parser.add_argument("--align_json", type=str, default=ALIGN_JSON)
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_query", type=int, default=32)
    parser.add_argument("--qformer_hidden", type=int, default=768)
    parser.add_argument("--qformer_layers", type=int, default=6)
    parser.add_argument("--contrast_dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def load_align_pairs(path: str, data_folder: str = None) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for item in data:
        video = item["video"]
        if isinstance(video, list):
            video = video[0]
        if data_folder:
            video = str(Path(data_folder) / video)
        if "caption" in item:
            text = item["caption"]
        elif "conversations" in item:
            text = next(
                (t["value"] for t in item["conversations"] if t.get("from") == "gpt"),
                "",
            )
        else:
            continue
        pairs.append({"video": video, "text": text})
    return pairs


class AlignDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def itc_loss(vis: torch.Tensor, txt: torch.Tensor, temperature: float) -> torch.Tensor:
    vis = F.normalize(vis, dim=-1)
    txt = F.normalize(txt, dim=-1)
    logits = (vis @ txt.t()) / temperature  # [B, B]
    labels = torch.arange(vis.shape[0], device=vis.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def main():
    args = parse_args()
    disable_torch_init()

    print(f"Loading visual tower from {args.qwen3vl_path} ...")
    qwen3vl = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen3vl_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    visual = qwen3vl.visual
    processor = AutoProcessor.from_pretrained(args.qwen3vl_path)

    print(f"Loading domain LLM from {args.llm_path} ...")
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    d_visual = getattr(visual.config, "hidden_size", None) or visual.config.embed_dim
    qformer_config = QFormerConfig(
        d_visual=d_visual,
        num_query=args.num_query,
        hidden_size=args.qformer_hidden,
        num_layers=args.qformer_layers,
    )
    model = MoTIFPlusForConditionalGeneration(visual, llm, qformer_config)

    for p in model.llm.parameters():
        p.requires_grad = False
    model.llm.eval()

    vis_proj = nn.Linear(args.qformer_hidden, args.contrast_dim)
    txt_proj = nn.Linear(llm.config.hidden_size, args.contrast_dim)

    trainable = list(model.qformer.parameters()) + \
                list(model.llm_proj.parameters()) + \
                [model.frame_embeddings] + \
                list(vis_proj.parameters()) + \
                list(txt_proj.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    pairs = load_align_pairs(args.align_json, data_folder=args.data_folder)
    print(f"Align pairs: {len(pairs)}")
    dataset = AlignDataset(pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    vis_proj.to(device)
    txt_proj.to(device)

    model.train()

    for epoch in range(args.num_epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.num_epochs}")
        for batch in pbar:
            video_paths = batch["video"]
            texts = batch["text"]

            proc = processor(videos=video_paths, return_tensors="pt")
            pixel_values = (
                proc["pixel_values_videos"]
                if "pixel_values_videos" in proc
                else proc["pixel_values"]
            ).to(device)

            query = model.forward_qformer(pixel_values)          # [B, nq, d_q]
            vis_emb = vis_proj(query.mean(dim=1))                # [B, contrast]

            text_tok = tokenizer(
                texts, padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(device)
            text_emb = model.llm.get_input_embeddings()(text_tok.input_ids)  # [B, L, d_llm]
            mask = text_tok.attention_mask.unsqueeze(-1).float()
            txt_emb = txt_proj((text_emb * mask).sum(1) / mask.sum(1).clamp(min=1))

            loss = itc_loss(vis_emb, txt_emb, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "qformer": model.qformer.state_dict(),
            "llm_proj": model.llm_proj.state_dict(),
            "frame_embeddings": model.frame_embeddings.data,
            "qformer_config": qformer_config,
        },
        out / "qformer.pt",
    )
    print(f"Stage-2 done. Q-Former saved to {out / 'qformer.pt'}")


if __name__ == "__main__":
    main()
