from pathlib import Path

from pytubefix import YouTube


def download_audio(yt_id: str, folder: Path) -> None:
    url = f"https://www.youtube.com/watch?v={yt_id}"
    yt = YouTube(url)

    def itags():
        max_audio = 0
        audio_value = 0
        for audio_stream in yt.streams.filter(only_audio=True):
            abr = int(audio_stream.abr.replace("kbps", ""))
            if abr > max_audio:
                max_audio = abr
                audio_value = audio_stream.itag
        return audio_value

    audio = itags()
    asd = yt.streams.get_by_itag(audio)
    asd.download(output_path=folder, filename=f"{yt_id}.m4a")
