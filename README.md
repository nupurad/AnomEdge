# Gemma-3n Factory Anomaly Classifier (JSON SFT)

Fine-tune Gemma-3n to output strict JSON for:
- `normal`
- `smoke_fire`
- `oil_leak`
- `conveyor_jam`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Download datasets

```bash
python -m src.data.download_dataset
```

Downloads:
- `neurobotdata/fire-and-smoke-in-confined-space-synthetic-dataset`
- `vighneshanand/oil-spill-dataset-binary-image-classification`
- `chiaravaliante/conveyor-belts` (normal conveyor data)

## 2) Add your 3 conveyor-jam anomaly images

Save your 3 images locally, then run:

```bash
python -m src.data.add_conveyor_jam_images \
  --images /absolute/path/jam1.jpg /absolute/path/jam2.jpg /absolute/path/jam3.jpg
```

This copies them into `data/raw/conveyor_jam/`.

## 3) Build classification folders

```bash
python -m src.data.prepare_dataset \
  --fire-smoke-root "/Users/nupurdashputre/.cache/kagglehub/datasets/neurobotdata/fire-and-smoke-in-confined-space-synthetic-dataset/versions/1" \
  --oil-binary-root "/Users/nupurdashputre/.cache/kagglehub/datasets/vighneshanand/oil-spill-dataset-binary-image-classification/versions/1" \
  --conveyor-normal-root "/Users/nupurdashputre/.cache/kagglehub/datasets/chiaravaliante/conveyor-belts/versions/1" \
  --conveyor-jam-root "data/raw/conveyor_jam" \
  --out-root "data/processed" \
  --val-ratio 0.2
```

## 4) Build `train.jsonl` / `eval.jsonl` for your SFT format

```bash
python -m src.data.build_jsonl \
  --data-root data/processed \
  --train-out src/train.jsonl \
  --eval-out src/eval.jsonl \
  --eval-ratio 0.2
```

Each JSONL row includes:
- `image`
- `frame_id`
- `timestamp`
- `anomaly_type`
- `confidence`
- `flags`
- `evidence`

## 5) Fine-tune Gemma-3n (your requested SFT style)

```bash
python -m src.train_gemma3n \
  --model-id google/gemma-3n-e2b-it \
  --train-jsonl src/train.jsonl \
  --eval-jsonl src/eval.jsonl \
  --output-dir models/gemma3n-json-lora \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 2e-4 \
  --epochs 2 \
  --eval-steps 200 \
  --save-steps 200
```

This follows the same pattern you gave: `load_dataset(json)`, `AutoProcessor`, LoRA via PEFT, and `SFTTrainer` over a combined `text` field.

## Output JSON schema

```json
{
  "frame_id": "string",
  "timestamp": 1700000000,
  "anomaly_type": "normal | smoke_fire | oil_leak | conveyor_jam",
  "confidence": 0.0,
  "flags": {
    "injury_risk": false,
    "is_spreading": false,
    "hazard_suspected": false,
    "conveyor_halted": false,
    "motor_overheating": false,
    "belt_damage_visible": false
  },
  "evidence": {
    "observations": ["short text descriptions"],
    "bbox": [
      { "label": "smoke", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4 }
    ]
  }
}
```
