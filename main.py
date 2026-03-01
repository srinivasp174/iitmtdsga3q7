import os
import time
import tempfile
import shutil
import re
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import yt_dlp
import google.genai as genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    video_url: str
    topic: str

class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str

def validate_timestamp(ts: str) -> bool:
    pattern = r"^\d{2}:\d{2}:\d{2}$"
    return bool(re.match(pattern, ts))

MIME_TYPE_MAP = {
    "m4a": "audio/mp4",
    "webm": "audio/webm",
    "mp3": "audio/mpeg",
    "mp4": "audio/mp4",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "opus": "audio/opus",
}

def download_audio(video_url: str) -> tuple[str, str, str]:
    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        file_path = ydl.prepare_filename(info)
        ext = info.get("ext", "").lower()

    mime_type = MIME_TYPE_MAP.get(ext)
    if not mime_type:
        raise Exception(f"Unsupported audio format: {ext}")

    return file_path, temp_dir, mime_type

def upload_and_wait(file_path: str, mime_type: str):
    uploaded_file = client.files.upload(
        file=file_path,
        config={"mime_type": mime_type}
    )

    for _ in range(30):
        current = client.files.get(name=uploaded_file.name)
        if current.state.name == "ACTIVE":
            return current
        elif current.state.name == "FAILED":
            raise Exception("Gemini failed to process the audio file.")
        time.sleep(2)

    raise Exception("Timed out waiting for Gemini to process the audio file.")

def find_timestamp(file, topic: str) -> str:
    prompt = f"""Listen to this audio carefully.

Find the FIRST moment when the topic "{topic}" is discussed or mentioned.

Respond with ONLY a JSON object like this:
{{"timestamp": "HH:MM:SS"}}

No markdown, no explanation, no code blocks. Just the JSON."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=file.uri,
                        mime_type=file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ]
            )
        ]
        # No response_mime_type — causes 500 with audio input
    )

    raw = response.text.strip()

    # Strip markdown code blocks if model added them anyway
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    if "timestamp" not in data:
        raise Exception(f"Model did not return a timestamp. Got: {raw}")

    return data["timestamp"]

@app.post("/ask", response_model=AskResponse)
def ask(data: AskRequest):
    temp_dir = None

    try:
        audio_path, temp_dir, mime_type = download_audio(data.video_url)
        uploaded_file = upload_and_wait(audio_path, mime_type)
        timestamp = find_timestamp(uploaded_file, data.topic)

        if not validate_timestamp(timestamp):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid timestamp format returned by model: {timestamp}"
            )

        return AskResponse(
            timestamp=timestamp,
            video_url=data.video_url,
            topic=data.topic
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)