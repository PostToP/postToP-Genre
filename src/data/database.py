import json
import logging
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from data.fetch import download_audio

load_dotenv()

logger = logging.getLogger("experiment")


def get_connection() -> psycopg2.extensions.connection:
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
    )


def fetch_videos() -> list[tuple]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""SELECT
    yt_id,genres
FROM
    posttop.video v
    INNER JOIN posttop.genre_review gr ON v.id = gr.video_id;""")
    videos = cursor.fetchall()
    cursor.close()
    conn.close()
    return videos


def convert_postgres_videos_to_json(videos: list[tuple]) -> list[dict]:
    all_vids = []
    for video in videos:
        obj = {
            "yt_id": video[0],
            "genres": video[1],
        }
        all_vids.append(obj)
    return all_vids


def save_videos_to_json(
    videos: list[dict], filename: str = "dataset/videos.json"
) -> None:
    with open(filename, "w") as f:
        json.dump(videos, f, indent=4)


def main() -> None:
    videos = fetch_videos()
    if not videos:
        logger.error("No videos fetched from database")
        return
    for video in videos:
        print(f"Fetched video: {video}")
    video_json = convert_postgres_videos_to_json(videos)
    output_dir = "./dataset/audio/"
    os.makedirs(output_dir, exist_ok=True)
    outdir_path = Path(output_dir)
    for video in video_json:
        print(f"Downloaded video JSON: {video}")
        download_audio(video["yt_id"], outdir_path)
    save_videos_to_json(video_json)
    logger.debug(f"Saved {len(video_json)} videos to dataset/videos.json")


if __name__ == "__main__":
    main()
