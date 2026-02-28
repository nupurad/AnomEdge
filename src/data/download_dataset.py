from pathlib import Path

import kagglehub


def download_fire_smoke_dataset() -> Path:
    dataset_path = kagglehub.dataset_download(
        "neurobotdata/fire-and-smoke-in-confined-space-synthetic-dataset"
    )
    out = Path(dataset_path)
    print(f"Path to dataset files: {out}")
    return out


def download_oil_binary_dataset() -> Path:
    dataset_path = kagglehub.dataset_download(
        "vighneshanand/oil-spill-dataset-binary-image-classification"
    )
    out = Path(dataset_path)
    print(f"Path to oil dataset files: {out}")
    return out


def download_conveyor_normal_dataset() -> Path:
    dataset_path = kagglehub.dataset_download("chiaravaliante/conveyor-belts")
    out = Path(dataset_path)
    print(f"Path to conveyor dataset files: {out}")
    return out


if __name__ == "__main__":
    download_fire_smoke_dataset()
    download_oil_binary_dataset()
    download_conveyor_normal_dataset()
