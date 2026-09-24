import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_instruct_data(json_path: str, data_folder: str = None) -> list:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        video = item["video"]
        if isinstance(video, list):
            video = video[0]
        if data_folder:
            video = str(Path(data_folder) / video)
        question, answer = None, None
        for turn in item["conversations"]:
            if turn.get("from") == "human":
                question = turn.get("value", "")
            elif turn.get("from") == "gpt":
                answer = turn.get("value", "")
        if question is not None and answer is not None:
            samples.append({"video": video, "question": question, "answer": answer})
    return samples


def strip_media_tags(text: str) -> str:
    for tag in ("<video>", "<image>"):
        text = text.replace(tag, "")
    return text.strip()


def build_prompt(question: str) -> str:
    question = strip_media_tags(question)
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def tokenize_turn(tokenizer, question: str, answer: str, max_length: int):
    prompt_ids = tokenizer(
        build_prompt(question), add_special_tokens=False
    ).input_ids
    answer_ids = tokenizer(
        f"{answer}<|im_end|>", add_special_tokens=False
    ).input_ids

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return input_ids, labels


class InstructDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class InstructCollator:
    def __init__(self, tokenizer, processor, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def _video_pixels(self, video_paths: list):
        inputs = self.processor(videos=video_paths, return_tensors="pt")
        if "pixel_values_videos" in inputs:
            return inputs["pixel_values_videos"]
        return inputs["pixel_values"]

    def __call__(self, batch: list):
        video_paths = [b["video"] for b in batch]
        pixel_values = self._video_pixels(video_paths)

        input_ids_list, labels_list = [], []
        for b in batch:
            input_ids, labels = tokenize_turn(
                self.tokenizer, b["question"], b["answer"], self.max_length
            )
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        input_ids = self._pad(input_ids_list, self.tokenizer.pad_token_id)
        labels = self._pad(labels_list, -100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    @staticmethod
    def _pad(sequences: list, pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in sequences)
        padded = [
            s + [pad_value] * (max_len - len(s)) for s in sequences
        ]
        return torch.tensor(padded, dtype=torch.long)


def make_instruct_data_module(tokenizer, processor, json_path: str, data_folder: str = None, max_length: int = 2048):
    samples = load_instruct_data(json_path, data_folder=data_folder)
    dataset = InstructDataset(samples)
    collator = InstructCollator(tokenizer, processor, max_length)
    return dataset, collator
