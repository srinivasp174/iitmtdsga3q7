import os
import time
import tempfile
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import yt_dlp
import google.genai as genai

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()


# ---------------------------------------------------
# Request & Response Models
# ---------------------------------------------------

class AskRequest(BaseModel):
    video_url: str
    topic: str


class AskResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


# ---------------------------------------------------
# Utility: Validate HH:MM:SS format
# ---------------------------------------------------

def validate_timestamp(ts: str) -> bool:
    pattern = r"^\d{2}:\d{2}:\d{2}$"
    return bool(re.match(pattern, ts))


# ---------------------------------------------------
# Step 1: Download Audio Only
# ---------------------------------------------------

def download_audio(video_url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",   # audio only
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        file_path = ydl.prepare_filename(info)

    return file_path


# ---------------------------------------------------
# Step 2: Upload to Gemini Files API
# ---------------------------------------------------

def upload_and_wait(file_path: str):
    uploaded_file = genai.upload_file(path=file_path)

    # Poll until ACTIVE
    while uploaded_file.state.name != "ACTIVE":
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)

    return uploaded_file


# ---------------------------------------------------
# Step 3: Ask Gemini for Timestamp
# ---------------------------------------------------

def find_timestamp(file, topic: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")

    response = model.generate_content(
        [
            file,
            f"""
            You are analyzing audio.
            Find the FIRST time the topic below is spoken.

            Topic: "{topic}"

            Return ONLY a timestamp in HH:MM:SS format.
            Do not explain anything.
            """
        ],
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "pattern": r"^\d{2}:\d{2}:\d{2}$"
                    }
                },
                "required": ["timestamp"]
            }
        }
    )

    result = response.json()

    return result["timestamp"]


# ---------------------------------------------------
# FastAPI Endpoint
# ---------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(data: AskRequest):

    audio_path = None

    try:
        # 1. Download audio
        audio_path = download_audio(data.video_url)

        # 2. Upload and wait
        uploaded_file = upload_and_wait(audio_path)

        # 3. Ask Gemini
        timestamp = find_timestamp(uploaded_file, data.topic)

        # 4. Validate format
        if not validate_timestamp(timestamp):
            raise HTTPException(status_code=500, detail="Invalid timestamp format")

        return AskResponse(
            timestamp=timestamp,
            video_url=data.video_url,
            topic=data.topic
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 5. Cleanup temp file
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)