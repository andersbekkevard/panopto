from openai import OpenAI

API_KEY = "OPENAI_API_KEY"


audio_paths = [
    "C:\\Users\\Anders\\forelesninger\\styring\\audio\\TIØ4105 - 17.03.2025 - Økt 2.mp3"
]


# This method has hard-coded file path for "styring"
def transcribeAndSaveLecture(IN_PATH: str) -> None:
    lecture_name = IN_PATH.split("\\")[-1][:-3] + "txt"
    OUT_PATH = "C:\\Users\\Anders\\forelesninger\\styring\\text\\" + lecture_name

    client = OpenAI(api_key=API_KEY)

    with open(IN_PATH, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )

    with open(OUT_PATH, "w", encoding="utf-8") as text_file:
        text_file.write(transcription.text)


def main():
    for IN_PATH in audio_paths:
        transcribeAndSaveLecture(IN_PATH)
