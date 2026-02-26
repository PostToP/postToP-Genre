from pathlib import Path

import pandas as pd

from config.config import AUDIO_LENGTH
import librosa
import soundfile as sf
import torchaudio

TARGET_SR = 16000


def split_audio_into_chunks(file_path: Path):
    print(f"Loading audio file: {file_path}")

    waveform, sr = torchaudio.load(file_path)  # (channels, samples)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)
        sr = TARGET_SR

    waveform = waveform.squeeze(0)  # (samples,)

    chunk_size = int(AUDIO_LENGTH * sr)

    chunks = [
        waveform[i : i + chunk_size]
        for i in range(0, waveform.shape[0], chunk_size)
        if waveform[i : i + chunk_size].shape[0] == chunk_size
    ]

    return chunks, sr


def preprocess_dataset():
    dataset = pd.read_json("dataset/videos.json")
    rows_list = []
    for _, row in dataset.iterrows():
        file_path = Path("dataset/audio") / f"{row['yt_id']}.m4a"
        chunks, sr = split_audio_into_chunks(file_path)
        for chunk in chunks:
            rows_list.append(
                {
                    "yt_id": row["yt_id"],
                    "audio_chunks": chunk.tolist(),
                    "sample_rate": sr,
                    "genres": row["genres"],
                }
            )
    new_df = pd.DataFrame(rows_list)
    new_df.to_json("dataset/p2_dataset.json", index=False)
