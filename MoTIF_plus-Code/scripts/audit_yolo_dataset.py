#!/usr/bin/env python3

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_names(cfg):
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError("data yaml has no valid 'names' field")


def resolve(value, base):
    path = Path(value)
    return path if path.is_absolute() else (base / path)


def list_images(spec, base):
    if isinstance(spec, (list, tuple)):
        images = []
        for item in spec:
            images.extend(list_images(item, base))
        return images
    spec = resolve(spec, base)
    if spec.is_dir():
        return sorted(p for p in spec.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if spec.is_file() and spec.suffix.lower() == ".txt":
        images = []
        for line in spec.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                images.append(resolve(line, spec.parent))
        return images
    return [spec]


def label_path(image, labels_root=None):
    labels_root = Path(labels_root) if labels_root is not None else None
    parts = list(image.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        rel_parts = parts[idx + 1:]
        rel = Path(*rel_parts) if rel_parts else Path(image.name)
        if labels_root is not None:
            return (labels_root / rel).with_suffix(".txt")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    if labels_root is not None:
        return (labels_root / image.name).with_suffix(".txt")
    return image.with_suffix(".txt")


def audit_split(images, names, labels_root):
    stats = defaultdict(lambda: {"frames": 0, "boxes": 0})
    qc = defaultdict(int)
    qc["images_total"] = len(images)
    for image in images:
        lp = label_path(image, labels_root)
        if not lp.exists():
            qc["images_missing_label"] += 1
            continue
        lines = [
            line.split()
            for line in lp.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not lines:
            qc["images_empty_label"] += 1
            continue
        present = set()
        seen = set()
        for parts in lines:
            if len(parts) < 5:
                qc["invalid_line"] += 1
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                qc["invalid_line"] += 1
                continue
            if cls not in names:
                qc["invalid_class_id"] += 1
                continue
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                    and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                qc["invalid_bbox"] += 1
                continue
            key = (cls, round(x, 4), round(y, 4), round(w, 4), round(h, 4))
            if key in seen:
                qc["duplicate_bbox"] += 1
            seen.add(key)
            stats[names[cls]]["boxes"] += 1
            present.add(names[cls])
        for name in present:
            stats[name]["frames"] += 1
    return stats, qc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO data yaml")
    parser.add_argument("--out", default="dataset_stats.csv", help="output CSV path")
    parser.add_argument(
        "--labels-root",
        default=None,
        help="optional labels root; default derives it by replacing images/ with labels/",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    cfg = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = load_names(cfg)
    base = data_path.parent

    split_specs = {k: cfg[k] for k in ("train", "val", "test") if k in cfg}
    if not split_specs:
        sys.exit("data yaml must contain train/val/test")

    rows = []
    for split, spec in split_specs.items():
        images = list_images(spec, base)
        stats, qc = audit_split(images, names, args.labels_root)
        print(
            f"[{split}] images={qc['images_total']} "
            f"missing_label={qc['images_missing_label']} "
            f"empty_label={qc['images_empty_label']} "
            f"invalid_class={qc['invalid_class_id']} "
            f"invalid_bbox={qc['invalid_bbox']} "
            f"duplicate_bbox={qc['duplicate_bbox']} "
            f"invalid_line={qc['invalid_line']}"
        )
        for cls in names.values():
            item = stats.get(cls, {"frames": 0, "boxes": 0})
            rows.append([split, cls, item["frames"], item["boxes"]])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "annotated_frames", "bboxes"])
        writer.writerows(rows)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
