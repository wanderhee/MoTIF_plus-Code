import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from motif_plus import QFormerConfig, MoTIFPlusForConditionalGeneration, disable_torch_init
from data.instruct_dataset import make_instruct_data_module


QWEN3VL_PATH = "Qwen/Qwen3-VL-8B-Instruct"
LLM_PATH = "./output/stage1/final_model"
QFORMER_CKPT = "./output/stage2/qformer.pt"
INSTRUCT_JSON = "./datasets/videomodified.json"
DATA_FOLDER = "./datasets"
OUTPUT_DIR = "./output/stage3"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen3vl_path", type=str, default=QWEN3VL_PATH)
    parser.add_argument("--llm_path", type=str, default=LLM_PATH)
    parser.add_argument("--qformer_ckpt", type=str, default=QFORMER_CKPT)
    parser.add_argument("--instruct_json", type=str, default=INSTRUCT_JSON)
    parser.add_argument("--data_folder", type=str, default=DATA_FOLDER)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_query", type=int, default=32)
    parser.add_argument("--qformer_hidden", type=int, default=768)
    parser.add_argument("--qformer_layers", type=int, default=6)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


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

    print(f"Loading LLM in 4-bit NF4 from {args.llm_path} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        quantization_config=bnb_config,
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

    print(f"Loading Q-Former from {args.qformer_ckpt} ...")
    ckpt = torch.load(args.qformer_ckpt, map_location="cpu")
    model.qformer.load_state_dict(ckpt["qformer"])
    model.llm_proj.load_state_dict(ckpt["llm_proj"])
    model.frame_embeddings.data.copy_(ckpt["frame_embeddings"])

    for p in model.qformer.parameters():
        p.requires_grad = False
    model.llm_proj.requires_grad_(False)
    model.frame_embeddings.requires_grad_(False)

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model.llm = prepare_model_for_kbit_training(model.llm)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model.llm = get_peft_model(model.llm, lora_config)

    dataset, collator = make_instruct_data_module(
        tokenizer, processor, args.instruct_json,
        data_folder=args.data_folder, max_length=args.max_length,
    )
    print(f"Train samples: {len(dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False,
        report_to="tensorboard",
        run_name=Path(args.output_dir).name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    print("Starting training ...")
    trainer.train()

    final_dir = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Stage-3 done. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
