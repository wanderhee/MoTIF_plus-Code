import argparse
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.pretrain_dataset import build_pretrain_dataset


MODEL_PATH = "Qwen/Qwen3-8B"
TEXT_DIR = "./datasets/pretrain_text"
OUTPUT_DIR = "./output/stage1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=MODEL_PATH)
    parser.add_argument("--text_dir", type=str, default=TEXT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--deepspeed", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model/tokenizer from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=True,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()

    print(f"Building dataset from {args.text_dir} ...")
    dataset = build_pretrain_dataset(args.text_dir)

    def tokenize(examples):
        tok = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="tokenizing",
    )

    print(f"Train samples: {len(tokenized['train'])}, valid: {len(tokenized['validation'])}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
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
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )
    print("Starting training ...")
    trainer.train()

    final_dir = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Stage-1 done. Domain LLM saved to {final_dir}")


if __name__ == "__main__":
    main()
