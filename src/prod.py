import tempfile
import traceback
from pathlib import Path
from typing import Dict
import subprocess

from flask import Flask, jsonify, request
from flask_cors import CORS

import numpy as np
import torchaudio

from config.config import SAMPLE_RATE, AUDIO_LENGTH, TABLE_BACK

from data.fetch import download_audio
from model.ModelWrapper import ModelWrapper

model_wrapper = ModelWrapper.deserialize("model/compiled_model.tar.gz")
model_wrapper.warmup()


def convert_audio_format(input_path: Path, output_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",  # mono
            "-y",  # overwrite
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def load_and_preprocess_audio(audio_path: Path) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    waveform_np = np.asarray(
        waveform.squeeze().detach().cpu().tolist(), dtype=np.float32
    )

    max_length = SAMPLE_RATE * AUDIO_LENGTH
    if len(waveform_np) > max_length:
        waveform_np = waveform_np[:max_length]

    return waveform_np


def predict_genres(yt_id: str, duration: float = 65.0) -> Dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        download_audio(yt_id, tmp_path, duration=duration)
        wav_path = tmp_path / f"{yt_id}.wav"
        convert_audio_format(tmp_path / f"{yt_id}.m4a", wav_path)

        waveform = load_and_preprocess_audio(wav_path)

        chunk_length = SAMPLE_RATE * 10
        chunks = [
            waveform[i : i + chunk_length]
            for i in range(0, len(waveform), chunk_length)
        ]

        all_logits = []
        for chunk in chunks:
            logits = model_wrapper.predict(chunk)
            all_logits.append(logits)

        aggregated_logits = np.mean(all_logits, axis=0)
        aggregated_logits = np.exp(aggregated_logits) / np.sum(
            np.exp(aggregated_logits)
        )
        print(f"Aggregated logits for {yt_id}: {aggregated_logits}")

        predicted_genre_id = int(np.argmax(aggregated_logits))
        predicted_genre_name = TABLE_BACK.get(
            predicted_genre_id, f"Genre_{predicted_genre_id}"
        )
        print(
            f"Predicted genre for {yt_id}: {predicted_genre_name} (ID: {predicted_genre_id})"
        )

        return {
            "yt_id": yt_id,
            "predicted_genres": [predicted_genre_name],
            "aggregated_logits": {
                TABLE_BACK[i]: float(logit)
                for i, logit in enumerate(aggregated_logits.flat)
            },
        }


app = Flask(__name__)
CORS(app)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        yt_id = data.get("yt_id")
        duration = data.get("duration")

        if not yt_id:
            return jsonify({"error": "yt_id is required"}), 400

        result = predict_genres(yt_id, duration=duration)

        return jsonify(
            {
                "prediction": result,
                "version": "1.0.0",
            }
        )
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    from waitress import serve

    print("Starting server on http://0.0.0.0:5000")
    serve(app, host="0.0.0.0", port=5000)
