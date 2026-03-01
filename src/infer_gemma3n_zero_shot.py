import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ALLOWED_ANOMALIES = {"normal", "smoke_fire", "oil_leak", "conveyor_jam"}

PROMPT = (
    "You are an industrial safety perception model.\n"
    "Analyze the image and output ONLY valid JSON, no markdown, no explanation.\n"
    "Use exactly this schema:\n"
    '{'
    '"frame_id":"string",'
    '"timestamp":1700000000,'
    '"anomaly_type":"normal|smoke_fire|oil_leak|conveyor_jam",'
    '"confidence":0.0,'
    '"flags":{'
    '"injury_risk":false,'
    '"is_spreading":false,'
    '"hazard_suspected":false,'
    '"conveyor_halted":false,'
    '"motor_overheating":false,'
    '"belt_damage_visible":false'
    "},"
    '"evidence":{'
    '"observations":["short evidence"],'
    '"bbox":[{"label":"smoke","x":0.1,"y":0.2,"w":0.3,"h":0.4}]'
    "}"
    "}\n"
    "Rules:\n"
    "- anomaly_type must be one of the 4 allowed values.\n"
    "- confidence must be a float in [0,1].\n"
    "- bbox can be empty list for normal.\n"
    "- Output exactly one JSON object.\n"
)


def device_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def to_device(batch: Dict[str, Any], device: torch.device, model_dtype: torch.dtype) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                out[k] = v.to(device=device, dtype=model_dtype)
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return text[start : end + 1]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _as_float_01(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, f))


def normalize_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    anomaly_type = str(obj.get("anomaly_type", "normal")).strip().lower()
    if anomaly_type not in ALLOWED_ANOMALIES:
        raise ValueError(f"Invalid anomaly_type: {anomaly_type}")

    flags = obj.get("flags", {}) if isinstance(obj.get("flags"), dict) else {}
    evidence = obj.get("evidence", {}) if isinstance(obj.get("evidence"), dict) else {}

    observations = evidence.get("observations", [])
    if not isinstance(observations, list):
        observations = [str(observations)]
    observations = [str(x) for x in observations][:8]

    bbox = evidence.get("bbox", [])
    if not isinstance(bbox, list):
        bbox = []
    cleaned_bbox = []
    for item in bbox[:16]:
        if not isinstance(item, dict):
            continue
        cleaned_bbox.append(
            {
                "label": str(item.get("label", "region")),
                "x": float(item.get("x", 0.0)),
                "y": float(item.get("y", 0.0)),
                "w": float(item.get("w", 0.0)),
                "h": float(item.get("h", 0.0)),
            }
        )

    return {
        "frame_id": str(obj.get("frame_id", f"frame_{uuid4().hex[:12]}")),
        "timestamp": int(obj.get("timestamp", int(time.time()))),
        "anomaly_type": anomaly_type,
        "confidence": _as_float_01(obj.get("confidence", 0.0)),
        "flags": {
            "injury_risk": _as_bool(flags.get("injury_risk", False)),
            "is_spreading": _as_bool(flags.get("is_spreading", False)),
            "hazard_suspected": _as_bool(flags.get("hazard_suspected", False)),
            "conveyor_halted": _as_bool(flags.get("conveyor_halted", False)),
            "motor_overheating": _as_bool(flags.get("motor_overheating", False)),
            "belt_damage_visible": _as_bool(flags.get("belt_damage_visible", False)),
        },
        "evidence": {
            "observations": observations,
            "bbox": cleaned_bbox,
        },
    }


def generate_json_once(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], images=[image], return_tensors="pt")
    enc = to_device(enc, device=device, model_dtype=model.dtype)

    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    prompt_len = enc["input_ids"].shape[1]
    new_tokens = out_ids[0][prompt_len:]
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
    json_text = extract_json_object(raw)
    parsed = json.loads(json_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON is not an object.")
    return normalize_result(parsed)


def infer_with_retries(
    model,
    processor,
    image: Image.Image,
    device: torch.device,
    retries: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    last_err = None
    for _ in range(retries):
        try:
            return generate_json_once(model, processor, image, device, max_new_tokens=max_new_tokens)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to produce valid JSON after {retries} attempts: {last_err}")


def resolve_image_path(image_path: Path) -> Path:
    candidates = [image_path]
    cwd = Path.cwd()
    raw = str(image_path)
    if raw.startswith("/data/"):
        candidates.append(cwd / raw.lstrip("/"))
    if not image_path.is_absolute():
        candidates.append((cwd / image_path).resolve())
    for p in candidates:
        if p.exists():
            return p
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Image file not found. Tried:\n{tried}")


def load_image(image_path: Path | None, camera_index: int) -> Image.Image:
    if image_path is not None:
        resolved = resolve_image_path(image_path)
        return Image.open(resolved).convert("RGB")
    cap = cv2.VideoCapture(camera_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to capture frame from camera.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot Gemma-3n image-to-anomaly-JSON inference.")
    parser.add_argument("--model-id", type=str, default="google/gemma-3n-E2B-it")
    parser.add_argument("--image", type=Path, default=None, help="Path to image file. If omitted, webcam is used.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=device_dtype(),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    image = load_image(args.image, args.camera_index)
    result = infer_with_retries(
        model=model,
        processor=processor,
        image=image,
        device=device,
        retries=args.retries,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
