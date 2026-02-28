import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List

CLASSES = ["normal", "smoke_fire", "oil_leak", "conveyor_jam"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: Path) -> List[Path]:
    items: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            items.append(p)
    return items


def flags_for_class(anomaly_type: str) -> Dict[str, bool]:
    flags = {
        "injury_risk": False,
        "is_spreading": False,
        "hazard_suspected": False,
        "conveyor_halted": False,
        "motor_overheating": False,
        "belt_damage_visible": False,
    }
    if anomaly_type == "smoke_fire":
        flags.update({"injury_risk": True, "is_spreading": True, "hazard_suspected": True})
    elif anomaly_type == "oil_leak":
        flags.update({"injury_risk": True, "hazard_suspected": True})
    elif anomaly_type == "conveyor_jam":
        flags.update(
            {
                "injury_risk": True,
                "conveyor_halted": True,
                "motor_overheating": True,
                "belt_damage_visible": True,
            }
        )
    return flags


def bbox_for_class(anomaly_type: str):
    if anomaly_type == "smoke_fire":
        return [{"label": "smoke", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}]
    if anomaly_type == "oil_leak":
        return [{"label": "oil", "x": 0.15, "y": 0.55, "w": 0.25, "h": 0.2}]
    if anomaly_type == "conveyor_jam":
        return [{"label": "jam", "x": 0.45, "y": 0.35, "w": 0.3, "h": 0.25}]
    return []


def observation_for_class(anomaly_type: str) -> str:
    if anomaly_type == "normal":
        return "No anomaly pattern visible."
    if anomaly_type == "smoke_fire":
        return "Visible smoke/fire indicators in scene."
    if anomaly_type == "oil_leak":
        return "Possible oil spill / leak region visible."
    return "Conveyor obstruction/jam visible on belt."


def build_record(image_path: Path, anomaly_type: str, ts: int) -> Dict:
    return {
        "image": str(image_path),
        "frame_id": image_path.stem,
        "timestamp": ts,
        "anomaly_type": anomaly_type,
        "confidence": 1.0,
        "flags": flags_for_class(anomaly_type),
        "evidence": {
            "observations": [observation_for_class(anomaly_type)],
            "bbox": bbox_for_class(anomaly_type),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train.jsonl/eval.jsonl for Gemma-3n JSON SFT.")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-out", type=Path, default=Path("src/train.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("src/eval.jsonl"))
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_rows: List[Dict] = []
    ts = int(time.time())

    for anomaly_type in CLASSES:
        class_dir = args.data_root / "train" / anomaly_type
        alt_dir = args.data_root / anomaly_type
        source_dir = class_dir if class_dir.exists() else alt_dir
        if not source_dir.exists():
            continue

        images = find_images(source_dir)
        random.shuffle(images)
        for img in images:
            all_rows.append(build_record(img, anomaly_type, ts))
            ts += 1

    if not all_rows:
        raise ValueError("No images found to build JSONL. Check --data-root.")

    random.shuffle(all_rows)
    split = int(len(all_rows) * (1.0 - args.eval_ratio))
    train_rows = all_rows[:split]
    eval_rows = all_rows[split:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.eval_out.parent.mkdir(parents=True, exist_ok=True)

    with args.train_out.open("w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with args.eval_out.open("w", encoding="utf-8") as f:
        for row in eval_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_rows)} train rows -> {args.train_out}")
    print(f"Wrote {len(eval_rows)} eval rows -> {args.eval_out}")


if __name__ == "__main__":
    main()
