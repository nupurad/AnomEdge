import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ("normal", "smoke_fire", "oil_leak", "belt_damage")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXT


def find_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if is_image(p)])


def sha1_for_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def copy_images(image_paths: Iterable[Path], out_dir: Path, prefix: str) -> List[Path]:
    ensure_dir(out_dir)
    written: List[Path] = []
    for i, src in enumerate(image_paths):
        dst = out_dir / f"{prefix}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        written.append(dst)
    return written


def split_train_val(images: List[Path], val_ratio: float, seed: int) -> Dict[str, List[Path]]:
    shuffled = images[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return {"val": shuffled[:n_val], "train": shuffled[n_val:]}


def maybe_downsample(images: List[Path], max_count: int | None, seed: int) -> List[Path]:
    if max_count is None or max_count <= 0 or len(images) <= max_count:
        return images
    sampled = images[:]
    rng = random.Random(seed)
    rng.shuffle(sampled)
    return sorted(sampled[:max_count])


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_conveyor_label(raw_label: str) -> str | None:
    label = (raw_label or "").strip().lower()
    if label == "good":
        return "normal"
    if label in {"tear", "wear"}:
        return "belt_damage"
    return None


def load_conveyor_annotated(raw_root: Path) -> Dict[str, List[Path]]:
    dataset_dir = raw_root / "Conveyer belt.paligemma" / "dataset"
    annotations_path = dataset_dir / "_annotations.train.jsonl"
    class_to_paths: Dict[str, List[Path]] = {klass: [] for klass in CLASSES}

    for row in read_jsonl(annotations_path):
        image_name = row.get("image")
        if not image_name:
            continue
        image_path = dataset_dir / image_name
        if not is_image(image_path):
            continue

        suffix = str(row.get("suffix", "")).strip()
        raw_label = suffix.split()[-1] if suffix else ""
        mapped = normalize_conveyor_label(raw_label)
        if mapped is None:
            continue
        class_to_paths[mapped].append(image_path)

    return class_to_paths


def load_belt_damage_extra(raw_root: Path) -> List[Path]:
    return find_images(raw_root / "belt_damage_extra")


def load_spills_dataset(raw_root: Path) -> List[Path]:
    return find_images(raw_root / "spills.paligemma" / "dataset")


def load_fire_dataset(raw_root: Path) -> List[Path]:
    return find_images(raw_root / "fire-in-factory.paligemma" / "dataset")


def load_normal_dataset(raw_root: Path) -> List[Path]:
    return find_images(raw_root / "Normal")


def dedupe_class_to_paths(class_to_paths: Dict[str, List[Path]]) -> tuple[Dict[str, List[Path]], Dict[str, int]]:
    seen_hashes: set[str] = set()
    deduped: Dict[str, List[Path]] = {klass: [] for klass in class_to_paths}
    duplicate_counts: Dict[str, int] = {klass: 0 for klass in class_to_paths}

    for klass, paths in class_to_paths.items():
        for path in paths:
            digest = sha1_for_file(path)
            if digest in seen_hashes:
                duplicate_counts[klass] += 1
                continue
            seen_hashes.add(digest)
            deduped[klass].append(path)

    return deduped, duplicate_counts


def build_manifest_rows(
    split_to_written: Dict[str, Dict[str, List[Path]]],
    original_lookup: Dict[Path, Path],
) -> List[dict]:
    rows: List[dict] = []
    for split, class_map in split_to_written.items():
        for klass, written_paths in class_map.items():
            for written_path in written_paths:
                source_path = original_lookup[written_path]
                rows.append(
                    {
                        "split": split,
                        "label": klass,
                        "image": str(written_path),
                        "source_image": str(source_path),
                        "source_dataset": source_path.parts[1] if len(source_path.parts) > 1 else source_path.parent.name,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean and prepare data/processed/{train,val}/<class> from the current raw datasets. "
            "Classes: normal|smoke_fire|oil_leak|belt_damage."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional global cap applied to each class after deduplication.",
    )
    parser.add_argument(
        "--max-oil-leak",
        type=int,
        default=None,
        help="Optional cap applied specifically to the oil_leak class after deduplication.",
    )
    parser.add_argument(
        "--include-conveyor-good-as-normal",
        action="store_true",
        help="Include conveyor 'Good' images in the normal class in addition to data/raw/Normal.",
    )
    args = parser.parse_args()

    class_to_paths: Dict[str, List[Path]] = {klass: [] for klass in CLASSES}

    class_to_paths["smoke_fire"].extend(load_fire_dataset(args.raw_root))
    class_to_paths["oil_leak"].extend(load_spills_dataset(args.raw_root))
    class_to_paths["normal"].extend(load_normal_dataset(args.raw_root))

    conveyor_classes = load_conveyor_annotated(args.raw_root)
    class_to_paths["belt_damage"].extend(conveyor_classes["belt_damage"])
    class_to_paths["belt_damage"].extend(load_belt_damage_extra(args.raw_root))
    if args.include_conveyor_good_as_normal:
        class_to_paths["normal"].extend(conveyor_classes["normal"])

    class_to_paths, duplicate_counts = dedupe_class_to_paths(class_to_paths)

    missing = [klass for klass, paths in class_to_paths.items() if not paths]
    if missing:
        raise ValueError(
            "No images found for required classes: "
            + ", ".join(missing)
            + ". Check data/raw contents before preparing the dataset."
        )

    class_to_paths = {
        klass: maybe_downsample(paths, args.max_per_class, args.seed)
        for klass, paths in class_to_paths.items()
    }
    class_to_paths["oil_leak"] = maybe_downsample(class_to_paths["oil_leak"], args.max_oil_leak, args.seed)

    if args.out_root.exists():
        shutil.rmtree(args.out_root)

    manifest_lookup: Dict[Path, Path] = {}
    split_to_written: Dict[str, Dict[str, List[Path]]] = {"train": {}, "val": {}}
    summary = {"classes": {}, "duplicate_counts": duplicate_counts}

    for klass, images in class_to_paths.items():
        split = split_train_val(images, args.val_ratio, args.seed)
        train_written = copy_images(split["train"], args.out_root / "train" / klass, prefix=f"{klass}_train")
        val_written = copy_images(split["val"], args.out_root / "val" / klass, prefix=f"{klass}_val")

        for src, dst in zip(split["train"], train_written):
            manifest_lookup[dst] = src
        for src, dst in zip(split["val"], val_written):
            manifest_lookup[dst] = src

        split_to_written["train"][klass] = train_written
        split_to_written["val"][klass] = val_written
        summary["classes"][klass] = {
            "total": len(images),
            "train": len(train_written),
            "val": len(val_written),
        }

    manifest_rows = build_manifest_rows(split_to_written, manifest_lookup)
    manifest_path = args.out_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.out_root / "summary.json"
    summary["raw_root"] = str(args.raw_root)
    summary["manifest"] = str(manifest_path)
    summary["include_conveyor_good_as_normal"] = args.include_conveyor_good_as_normal
    summary["max_per_class"] = args.max_per_class
    summary["max_oil_leak"] = args.max_oil_leak
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared cleaned dataset at: {args.out_root.resolve()}")
    for klass in CLASSES:
        info = summary["classes"][klass]
        print(
            f"{klass}: total={info['total']}, train={info['train']}, val={info['val']}, "
            f"duplicates_skipped={duplicate_counts[klass]}"
        )
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
