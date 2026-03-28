import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

CLASSES = ["normal", "smoke_fire", "oil_leak", "belt_damage"]
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
    elif anomaly_type == "belt_damage":
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
    if anomaly_type == "belt_damage":
        return [{"label": "belt_damage", "x": 0.45, "y": 0.35, "w": 0.3, "h": 0.25}]
    return []


def observation_for_class(anomaly_type: str) -> str:
    if anomaly_type == "normal":
        return "No anomaly pattern visible."
    if anomaly_type == "smoke_fire":
        return "Visible smoke/fire indicators in scene."
    if anomaly_type == "oil_leak":
        return "Possible oil spill / leak region visible."
    return "Visible conveyor belt damage in scene."


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
    args = parser.parse_args()

    train_rows: List[Dict] = []
    eval_rows: List[Dict] = []
    ts = int(time.time())

    for anomaly_type in CLASSES:
        train_dir = args.data_root / "train" / anomaly_type
        val_dir = args.data_root / "val" / anomaly_type
        alt_dir = args.data_root / anomaly_type

        if train_dir.exists():
            for img in find_images(train_dir):
                train_rows.append(build_record(img, anomaly_type, ts))
                ts += 1
        elif alt_dir.exists():
            for img in find_images(alt_dir):
                train_rows.append(build_record(img, anomaly_type, ts))
                ts += 1

        if val_dir.exists():
            for img in find_images(val_dir):
                eval_rows.append(build_record(img, anomaly_type, ts))
                ts += 1

    if not train_rows:
        raise ValueError("No train images found to build JSONL. Check --data-root.")

    if not eval_rows:
        raise ValueError(
            "No validation images found to build eval JSONL. "
            "Expected data/processed/val/<class> folders."
        )

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
