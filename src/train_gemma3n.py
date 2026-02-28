import argparse
import json
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments
from trl import SFTTrainer

PROMPT = (
    "You are an industrial safety perception model.\n"
    "Analyze the image and output ONLY valid JSON with this schema:\n"
    '{ "frame_id": string, "timestamp": int, "anomaly_type": "normal|smoke_fire|oil_leak|conveyor_jam", '
    '"confidence": float, "flags": {"injury_risk": bool, "is_spreading": bool, "hazard_suspected": bool, '
    '"conveyor_halted": bool, "motor_overheating": bool, "belt_damage_visible": bool}, '
    '"evidence": {"observations": string[], "bbox": object[]} }\n'
    "No extra text. Output JSON only.\n"
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
    return {
        "prompt": PROMPT,
        "completion": completion,
        "image": ex.get("image", ex.get("image_path", "")),
    }


def to_sft_text(ex):
    return {"text": ex["prompt"] + ex["completion"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma-3n LoRA SFT in JSON-output format.")
    parser.add_argument("--model-id", type=str, default="google/gemma-3n-e2b-it")
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
    ds = ds.map(to_sft_text)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        dataset_text_field="text",
        args=train_args,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
