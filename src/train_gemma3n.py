import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Image as HFImage
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

PROMPT = (
    "You are an industrial safety perception model.\n"
    "Analyze the image and output ONLY one valid JSON object.\n"
    "Do not output markdown, code fences, explanations, notes, reasoning, or extra text.\n"
    "Do not wrap the JSON in backticks.\n"
    "Do not invent anomalies that are not visually supported by the image.\n"
    "Do not guess hidden causes, machine state, or hazards unless they are visually evident.\n"
    "If the scene does not clearly show smoke/fire, oil leak, or belt damage, use anomaly_type=\"normal\".\n"
    "Use exactly this schema:\n"
    '{'
    '"frame_id": string, '
    '"timestamp": int, '
    '"anomaly_type": "normal|smoke_fire|oil_leak|belt_damage", '
    '"confidence": float, '
    '"flags": {'
    '"injury_risk": bool, '
    '"is_spreading": bool, '
    '"hazard_suspected": bool, '
    '"conveyor_halted": bool, '
    '"motor_overheating": bool, '
    '"belt_damage_visible": bool'
    '}, '
    '"evidence": {'
    '"observations": string[], '
    '"bbox": object[]'
    "}"
    '}\n'
    "Guardrails:\n"
    "- anomaly_type must be exactly one of: normal, smoke_fire, oil_leak, belt_damage.\n"
    "- confidence must be a float between 0.0 and 1.0.\n"
    "- observations must be short, image-grounded statements only.\n"
    "- bbox must be an empty list if no anomaly region is clearly visible.\n"
    "- bbox entries must describe only visible regions in the image.\n"
    "- Set belt_damage_visible=true only when belt tear, wear, or visible belt damage is present.\n"
    "- Set conveyor_halted=true only if stoppage is visually evident; otherwise false.\n"
    "- Set motor_overheating=true only if there is visible evidence such as smoke, glow, or clear overheating signs.\n"
    "- If uncertain, choose the safer conservative output and lower confidence rather than inventing details.\n"
    "Return JSON only.\n"
)



def format_example(ex):
    target = {
        "frame_id": ex["frame_id"],
        "timestamp": ex["timestamp"],
        "anomaly_type": ex["anomaly_type"],
        "confidence": ex.get("confidence", 0.0),
        "flags": ex["flags"],
        "evidence": ex.get("evidence", {"observations": [], "bbox": []}),
    }
    completion = json.dumps(target, separators=(",", ":"), ensure_ascii=False)

    img = ex.get("image", ex.get("image_path", ""))
    if isinstance(img, str) and img:
        img = str(Path(img).resolve())

    return {"prompt": PROMPT, "completion": completion, "image": img}


def to_sft_record(ex):
    # Keep `content` type consistent (list of blocks) for all messages.
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ex["completion"]},
                ],
            },
        ],
        "image": ex["image"],
    }


def _validate_image_paths(ds_split, split_name: str) -> None:
    missing = []
    for i, path in enumerate(ds_split["image"]):
        if not path or not Path(path).exists():
            missing.append((i, path))
        if len(missing) >= 5:
            break
    if missing:
        details = "\n".join([f"  - idx={idx}, image='{path}'" for idx, path in missing])
        raise FileNotFoundError(
            f"{split_name} split has missing image files. First {len(missing)} missing entries:\n{details}"
        )


def _ensure_rgb(example):
    image = example["image"]
    if hasattr(image, "convert"):
        example["image"] = image.convert("RGB")
    return example


def _build_sft_config(args) -> SFTConfig:
    sig = inspect.signature(SFTConfig.__init__).parameters
    kwargs = dict(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        report_to="none",
    )
    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig:
        kwargs["eval_strategy"] = "steps"
    return SFTConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma-3n LoRA SFT (vision + JSON output).")
    parser.add_argument("--model-id", type=str, default="google/gemma-3n-E2B-it")
    parser.add_argument("--train-jsonl", type=Path, default=Path("src/train.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("src/eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/gemma3n-json-lora"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    ds = load_dataset(
        "json",
        data_files={
            "train": str(args.train_jsonl),
            "eval": str(args.eval_jsonl),
        },
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    _ = processor

    ds = ds.map(format_example)
    original_cols = ds["train"].column_names
    ds = ds.map(to_sft_record, remove_columns=original_cols)

    _validate_image_paths(ds["train"], "train")
    _validate_image_paths(ds["eval"], "eval")
    ds = ds.cast_column("image", HFImage(decode=True))
    ds = ds.map(_ensure_rgb)

    model = AutoModelForImageTextToText.from_pretrained(args.model_id, torch_dtype="auto")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    sft_args = _build_sft_config(args)

    base_collator = DataCollatorForVisionLanguageModeling(processor=processor)

    def data_collator(examples):
        batch = base_collator(examples)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                batch[key] = value.to(dtype=model.dtype)
        return batch

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        processing_class=processor,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
