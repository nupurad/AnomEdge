import argparse
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy provided conveyor-jam anomaly images into data/raw/conveyor_jam."
    )
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        required=True,
        help="Absolute or relative paths to jam anomaly images.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw/conveyor_jam"),
        help="Destination folder for conveyor_jam samples.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for i, image_path in enumerate(args.images, start=1):
        if not image_path.exists():
            print(f"Skipping missing file: {image_path}")
            continue
        if image_path.suffix.lower() not in IMG_EXT:
            print(f"Skipping non-image file: {image_path}")
            continue

        dst = args.out_dir / f"conveyor_jam_user_{i:03d}{image_path.suffix.lower()}"
        shutil.copy2(image_path, dst)
        copied += 1
        print(f"Copied: {image_path} -> {dst}")

    print(f"Total copied conveyor-jam images: {copied}")


if __name__ == "__main__":
    main()
