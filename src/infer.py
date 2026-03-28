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

ALLOWED_ANOMALIES = {"normal", "smoke_fire", "oil_leak", "belt_damage"}

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
    if start == -1:
        snippet = text.strip()
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        raise ValueError(f"No JSON object found in model output. Raw output: {snippet!r}")
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(text[start:])
    return text[start : start + end]


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


def resolve_image_path(image_path: Path) -> Path:
    candidates = [image_path]
    cwd = Path.cwd()
    raw = str(image_path)
    if raw.startswith("/data/"):
        candidates.append(cwd / raw.lstrip("/"))
    if not image_path.is_absolute():
        candidates.append((cwd / image_path).resolve())
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(f"  - {path}" for path in candidates)
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


def load_model_and_processor(model_dir: Path, base_model: str):
    processor_source = str(model_dir) if (model_dir / "preprocessor_config.json").exists() else base_model
    processor = AutoProcessor.from_pretrained(processor_source)

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftModel

        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            dtype=device_dtype(),
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(model, str(model_dir))
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            dtype=device_dtype(),
            low_cpu_mem_usage=True,
        )

    model.eval()
    return model, processor


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
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Failed to produce valid JSON after {retries} attempts: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fine-tuned Gemma-3n image-to-anomaly-JSON inference.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/gemma3n-json-lora"))
    parser.add_argument("--base-model", type=str, default="google/gemma-3n-E2B-it")
    parser.add_argument("--image", type=Path, default=None, help="Path to image file. If omitted, webcam is used.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model, processor = load_model_and_processor(args.model_dir, args.base_model)
    model.to(device)

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
