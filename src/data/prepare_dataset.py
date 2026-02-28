import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            images.append(p)
    return images


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_images(image_paths: List[Path], out_dir: Path, prefix: str) -> None:
    ensure_dir(out_dir)
    for i, src in enumerate(image_paths):
        dst = out_dir / f"{prefix}_{i:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)


def split_train_val(images: List[Path], val_ratio: float, seed: int) -> Dict[str, List[Path]]:
    random.seed(seed)
    shuffled = images[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return {"val": shuffled[:n_val], "train": shuffled[n_val:]}


def classify_oil_binary_image_path(path: Path) -> str | None:
    text = str(path).lower().replace("-", "_").replace(" ", "_")
    # Check negative labels first; they still contain the token "oil".
    negative_tokens = ["no_oil", "non_oil", "without_oil", "nooil", "clean"]
    if any(token in text for token in negative_tokens):
        return "normal"
    if "oil" in text:
        return "oil_leak"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build data/processed/{train,val}/<class> folders for "
            "normal|smoke_fire|oil_leak|conveyor_jam."
        )
    )
    parser.add_argument(
        "--fire-smoke-root",
        type=Path,
        required=True,
        help="Path from kagglehub download for fire/smoke images.",
    )
    parser.add_argument(
        "--oil-binary-root",
        type=Path,
        default=None,
        help=(
            "Path to kaggle oil binary dataset root. "
            "Images in 'no_oil*' folders map to normal; 'oil*' map to oil_leak."
        ),
    )
    parser.add_argument(
        "--normal-root",
        type=Path,
        default=None,
        help="Folder with normal factory images.",
    )
    parser.add_argument(
        "--conveyor-normal-root",
        type=Path,
        default=None,
        help="Path to normal-only conveyor belt dataset; all images are mapped to normal.",
    )
    parser.add_argument(
        "--oil-leak-root",
        type=Path,
        default=None,
        help="Folder with oil leak images.",
    )
    parser.add_argument(
        "--conveyor-jam-root",
        type=Path,
        default=None,
        help="Folder with conveyor jam images.",
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("data/processed"), help="Output dataset root."
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-single-class",
        action="store_true",
        help="Allow preparing data with just one class (not suitable for standard classification training).",
    )
    args = parser.parse_args()

    class_to_paths = {"smoke_fire": find_images(args.fire_smoke_root)}

    if args.oil_binary_root is not None:
        oil_images = find_images(args.oil_binary_root)
        oil_pos = []
        oil_neg = []
        unknown = 0
        for image_path in oil_images:
            klass = classify_oil_binary_image_path(image_path)
            if klass == "oil_leak":
                oil_pos.append(image_path)
            elif klass == "normal":
                oil_neg.append(image_path)
            else:
                unknown += 1

        if oil_pos:
            class_to_paths["oil_leak"] = class_to_paths.get("oil_leak", []) + oil_pos
        if oil_neg:
            class_to_paths["normal"] = class_to_paths.get("normal", []) + oil_neg

        print(
            "Oil-binary mapping: "
            f"oil_leak={len(oil_pos)}, normal={len(oil_neg)}, skipped_unknown={unknown}"
        )

    optional_roots = {
        "normal": args.normal_root,
        "oil_leak": args.oil_leak_root,
        "conveyor_jam": args.conveyor_jam_root,
    }
    missing_optional = []
    empty_optional = []

    for class_name, root in optional_roots.items():
        if root is None:
            missing_optional.append(class_name)
            continue
        images = find_images(root)
        if images:
            class_to_paths[class_name] = class_to_paths.get(class_name, []) + images
        else:
            empty_optional.append((class_name, root))

    if args.conveyor_normal_root is not None:
        conveyor_normal_images = find_images(args.conveyor_normal_root)
        if conveyor_normal_images:
            class_to_paths["normal"] = class_to_paths.get("normal", []) + conveyor_normal_images
            print(
                "Conveyor-normal mapping: "
                f"normal={len(conveyor_normal_images)}"
            )
        else:
            empty_optional.append(("normal (from conveyor-normal-root)", args.conveyor_normal_root))

    if not class_to_paths["smoke_fire"]:
        raise ValueError(
            "No images found for class 'smoke_fire'. "
            "Check --fire-smoke-root and make sure it contains image files."
        )

    if empty_optional:
        details = ", ".join([f"{c} ({p})" for c, p in empty_optional])
        print(f"Warning: no images found for optional class roots: {details}")
    if missing_optional:
        print(f"Warning: optional class roots not provided: {', '.join(missing_optional)}")

    if len(class_to_paths) < 2 and not args.allow_single_class:
        raise ValueError(
            "Only one class is available (smoke_fire). "
            "Provide at least one additional class (normal/oil_leak/conveyor_jam), "
            "or pass --allow-single-class to just prepare files."
        )

    if args.out_root.exists():
        shutil.rmtree(args.out_root)

    for class_name, images in class_to_paths.items():
        split = split_train_val(images, args.val_ratio, args.seed)
        copy_images(
            split["train"],
            args.out_root / "train" / class_name,
            prefix=f"{class_name}_train",
        )
        copy_images(
            split["val"],
            args.out_root / "val" / class_name,
            prefix=f"{class_name}_val",
        )
        print(
            f"{class_name}: train={len(split['train'])}, "
            f"val={len(split['val'])}, total={len(images)}"
        )

    print(f"Prepared dataset at: {args.out_root.resolve()}")
    print(f"Classes included: {', '.join(sorted(class_to_paths.keys()))}")


if __name__ == "__main__":
    main()
