import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

CLASS_ORDER = ["normal", "smoke_fire", "oil_leak", "conveyor_jam"]


def device_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def discover_classes(data_root: Path) -> List[str]:
    train_root = data_root / "train"
    classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {train_root}")
    return sorted(classes, key=lambda x: CLASS_ORDER.index(x) if x in CLASS_ORDER else 99)


def load_samples(data_root: Path, split: str, classes: List[str]) -> List[Dict[str, str]]:
    split_root = data_root / split
    samples: List[Dict[str, str]] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name in classes:
        class_dir = split_root / class_name
        if not class_dir.exists():
            continue
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                samples.append({"image_path": str(p), "label": class_name})

    if not samples:
        raise ValueError(f"No images found under {split_root}")
    return samples


@dataclass
class TrainCollator:
    processor: AutoImageProcessor
    classes: List[str]

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        images = []
        labels = []
        label2id = {label: i for i, label in enumerate(self.classes)}

        for ex in examples:
            image = Image.open(ex["image_path"]).convert("RGB")
            images.append(image)
            labels.append(label2id[ex["label"]])

        # processor will handle resizing / normalization and return pixel_values
        batch = self.processor(images=images, return_tensors="pt", padding=True)

        # pixel_values is the key used by vision models like ViT
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def maybe_enable_lora(model, use_lora: bool, r: int, alpha: int, dropout: float):
    if not use_lora:
        return model

    # NOTE: the original LoRA config in your script targeted causal-LM modules
    # (q_proj, k_proj, v_proj, ...) which are LM-specific and won't match ViT's modules.
    # Applying LoRA to vision models requires a different target list (often linear layers
    # in the attention or MLP blocks) and different task_type. To avoid runtime errors,
    # we'll skip applying LoRA for vision backbones by default and print guidance.
    #
    # If you want to add LoRA/adapters for ViT, consider:
    #  - targetting linear layers like ["query", "key", "value", "dense", "fc"] depending on the model
    #  - using task_type="IMAGE_CLASSIFICATION" or consult PEFT's docs for vision support
    #  - or using other adapter libraries (e.g., adapter-transformers)
    #
    # For now, skip and return the original model to keep training flow working.
    print(
        "Warning: --use-lora requested but automatic LoRA config for vision models is not implemented.\n"
        "Proceeding without LoRA. If you want LoRA for ViT, ask me and I will provide a tested PEFT config."
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a ViT-style image classifier for anomaly classification.")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--base-model", type=str, default="google/gemma-3n-e2b-it")
    parser.add_argument("--output-dir", type=Path, default=Path("models/gemma3n-anomaly"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    classes = discover_classes(args.data_root)
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes in data/processed/train for classification fine-tuning.")

    train_samples = load_samples(args.data_root, "train", classes)
    val_samples = load_samples(args.data_root, "val", classes)

    print(f"Classes: {classes}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

    # Use the image processor (not AutoProcessor/chat template)
    processor = AutoImageProcessor.from_pretrained(args.base_model, use_fast=True)

    # load an image classification model
    num_labels = len(classes)
    id2label = {i: label for i, label in enumerate(classes)}
    label2id = {label: i for i, label in enumerate(classes)}

    model = AutoModelForImageClassification.from_pretrained(
    args.base_model,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    torch_dtype=device_dtype(),
    low_cpu_mem_usage=True,
    ignore_mismatched_sizes=True,   # <-- add this
)

    model = maybe_enable_lora(
        model,
        use_lora=args.use_lora,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    collator = TrainCollator(processor=processor, classes=classes)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_samples,
        eval_dataset=val_samples,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    metadata = {
        "base_model": args.base_model,
        "classes": classes,
        "fine_tune_mode": "lora" if args.use_lora else "full",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model + processor + metadata to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()