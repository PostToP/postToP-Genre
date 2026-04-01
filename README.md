# AI Genre Service

Compact backend that classifies YouTube audio into genres. Includes data prep CLI, training loop (AST fine-tune), ONNX export, and a Flask + Waitress inference API that pulls audio via a fetch proxy.

## Stack

- Python 3.11
- Flask + Waitress
- PyTorch + torchaudio
- Hugging Face AST (`MIT/ast-finetuned-audioset-10-10-0.4593`)
- NumPy, pandas, scikit-learn
- ONNX Runtime
- ffmpeg (runtime audio resample)

## Requirements

- Python 3.11+
- ffmpeg on PATH
- pip

## Environment

Create a `.env` in the repo root.

- `AUDIO_FETCH_URL` – PROXY HTTP endpoint used by `data.fetch.download_audio` (defaults to `http://localhost:5000/fetch_audio`)

## Run API

```bash
pip install -r requirements.txt
python src/prod.py
```

Service listens on `http://0.0.0.0:5000`.

## API

- `POST /predict`

Request body:

```json
{
  "yt_id": "YOUTUBE_VIDEO_ID",
  "duration": 65.0
}
```

Response includes `predicted_genres` and per-genre probabilities. Uses `model/compiled_model.tar.gz`.

## CLI Pipeline

Runs chained via `python src/cli.py <steps>`.

- `fetch` – build `dataset/p2_dataset.json` from `dataset/audio/` structure
- `preprocess` – normalize labels / metadata
- `split` – create train/val JSON splits
- `tokenize` – prepare audio chunk metadata for AST input
- `train` – fine-tune AST and save `model/final_model.pth`
- `compile` – export ONNX wrapper `model/compiled_model.tar.gz` and validate F1

Example full run:

```bash
python src/cli.py fetch preprocess split tokenize train compile
```

## Docker

```bash
docker build -t ai-genre .
docker run --rm -p 5000:5000 ai-genre
```

Or pull and run the published image:

```bash
docker pull ghcr.io/posttop/ai-genre:latest
docker run --rm -p 5000:5000 ghcr.io/posttop/ai-genre:latest
```