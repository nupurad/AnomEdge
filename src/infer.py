import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_CLASSES = ["normal", "smoke_fire", "oil_leak", "conveyor_jam"]


def device_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


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
        flags["injury_risk"] = True
        flags["is_spreading"] = True
        flags["hazard_suspected"] = True
    elif anomaly_type == "oil_leak":
        flags["injury_risk"] = True
        flags["hazard_suspected"] = True
    elif anomaly_type == "conveyor_jam":
        flags["conveyor_halted"] = True
        flags["motor_overheating"] = True
        flags["belt_damage_visible"] = True
    return flags


def observations_for_class(anomaly_type: str, conf: float) -> List[str]:
    if anomaly_type == "normal":
        return [f"No critical anomaly detected (confidence={conf:.2f})."]
    if anomaly_type == "smoke_fire":
        return [f"Smoke/fire-like visual pattern detected (confidence={conf:.2f})."]
    if anomaly_type == "oil_leak":
        return [f"Oil leak-like region detected (confidence={conf:.2f})."]
    return [f"Conveyor jam-like obstruction detected (confidence={conf:.2f})."]


def bbox_stub(anomaly_type: str) -> List[Dict[str, float]]:
    if anomaly_type == "smoke_fire":
        return [{"label": "smoke", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}]
    if anomaly_type == "oil_leak":
        return [{"label": "oil", "x": 0.15, "y": 0.55, "w": 0.25, "h": 0.2}]
    if anomaly_type == "conveyor_jam":
        return [{"label": "jam", "x": 0.45, "y": 0.35, "w": 0.3, "h": 0.25}]
    return []


def to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    model_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                out[k] = v.to(device=device, dtype=model_dtype)
            else:
                out[k] = v.to(device)
    return out


def prompt_text(classes: List[str]) -> str:
    return (
        "Classify this factory frame into exactly one label from: "
        + ", ".join(classes)
        + ". Return only the label text."
    )


def load_model_and_processor(model_dir: Path, base_model: str):
    metadata_path = model_dir / "metadata.json"
    classes = DEFAULT_CLASSES
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        classes = meta.get("classes", DEFAULT_CLASSES)

    processor_source = str(model_dir) if (model_dir / "preprocessor_config.json").exists() else base_model
    processor = AutoProcessor.from_pretrained(processor_source)

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=device_dtype(),
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(model_dir))
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            torch_dtype=device_dtype(),
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, processor, classes


def score_candidate(
    model,
    processor,
    image: Image.Image,
    classes: List[str],
    candidate: str,
    device: torch.device,
) -> float:
    user_message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text(classes)},
            ],
        }
    ]

    prompt = processor.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)
    full = prompt + candidate

    enc_prompt = processor(text=[prompt], images=[image], return_tensors="pt", padding=False)
    enc_full = processor(text=[full], images=[image], return_tensors="pt", padding=False)

    prompt_len = enc_prompt["input_ids"].shape[1]
    labels = enc_full["input_ids"].clone()
    labels[:, :prompt_len] = -100

    model_inputs = to_device(enc_full, device, next(model.parameters()).dtype)
    model_inputs["labels"] = labels.to(device)

    with torch.no_grad():
        out = model(**model_inputs)
    return -float(out.loss.item())


def predict_image(
    model,
    processor,
    classes: List[str],
    image_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    scores = [score_candidate(model, processor, image, classes, c, device) for c in classes]
    probs = torch.softmax(torch.tensor(scores), dim=0)
    best_idx = int(torch.argmax(probs).item())

    anomaly_type = classes[best_idx]
    confidence = float(probs[best_idx].item())

    return {
        "frame_id": f"frame_{uuid4().hex[:12]}",
        "timestamp": int(time.time()),
        "anomaly_type": anomaly_type,
        "confidence": round(confidence, 4),
        "flags": flags_for_class(anomaly_type),
        "evidence": {
            "observations": observations_for_class(anomaly_type, confidence),
            "bbox": bbox_stub(anomaly_type),
        },
    }


def predict_webcam(model, processor, classes: List[str], camera_index: int, device: torch.device) -> Dict[str, Any]:
    cap = cv2.VideoCapture(camera_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to capture frame from camera.")
    temp_path = Path("/tmp/anomedge_frame.jpg")
    cv2.imwrite(str(temp_path), frame)
    return predict_image(model, processor, classes, temp_path, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma-3n inference and print anomaly JSON.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/gemma3n-anomaly"))
    parser.add_argument("--base-model", type=str, default="google/gemma-3n-e2b-it")
    parser.add_argument("--image", type=Path, default=None, help="Path to image file.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index if --image not set.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, processor, classes = load_model_and_processor(args.model_dir, args.base_model)
    model.to(device)

    if args.image is not None:
        result = predict_image(model, processor, classes, args.image, device)
    else:
        result = predict_webcam(model, processor, classes, args.camera_index, device)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
