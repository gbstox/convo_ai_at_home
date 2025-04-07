import os
import json
import tempfile
import wave
import whisper
import torch
import asyncio
import httpx
import base64
import time
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model = whisper.load_model("base")

# --------- Set your Ollama model here ---------
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
print(f"Using Ollama model: {OLLAMA_MODEL}")
# ----------------------------------------------

ollama_client = OpenAI(
    base_url="http://ollama:11434/v1",
    api_key="ollama"
)

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("Missing HF_TOKEN environment variable")

segmentation_model = Model.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token=hf_token
)
vad_pipeline = VoiceActivityDetection(segmentation=segmentation_model)
vad_pipeline.instantiate({
    "min_duration_on": 0.3,
    "min_duration_off": 0.2
})

# --------- Ollama model auto-pull logic ---------
OLLAMA_URL = "http://ollama:11434"

async def ensure_model():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            existing = {m["name"] for m in tags}
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
            return

        if not any(OLLAMA_MODEL in m for m in existing):
            print(f"Model '{OLLAMA_MODEL}' not found, pulling...")
            try:
                pull_resp = await client.post(f"{OLLAMA_URL}/api/pull", json={"name": OLLAMA_MODEL})
                pull_resp.raise_for_status()
                print(f"Started pulling '{OLLAMA_MODEL}'")
            except Exception as e:
                print(f"Error pulling model '{OLLAMA_MODEL}': {e}")
        else:
            print(f"Model '{OLLAMA_MODEL}' already present.")

@app.on_event("startup")
async def startup_event():
    await ensure_model()

# --------- End Ollama model auto-pull ---------

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sample_rate = 16000
    bytes_per_sample = 2

    buffer = b""
    audio_buffer = b""
    chunk_duration = 0.6
    chunk_size = int(sample_rate * bytes_per_sample * chunk_duration)

    conversation_history = []

    interrupt_event = asyncio.Event()
    tts_task = None
    processing_task = None

    user_speaking = False
    silence_start = None
    silence_threshold = 0.3

    speech_counter = 0
    speech_confirm_threshold = 2

    async def send_status(msg):
        try:
            await websocket.send_text(json.dumps({"type": "status", "message": msg}))
        except:
            pass

    async def stream_tts(reply_text):
        nonlocal interrupt_event
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                url = "http://kokoro:8880/dev/captioned_speech"
                payload = {
                    "model": "kokoro",
                    "voice": "af_bella",
                    "input": reply_text,
                    "response_format": "mp3",
                    "stream": True
                }
                headers = {"Content-Type": "application/json"}
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    async for line in resp.aiter_lines():
                        if interrupt_event.is_set():
                            print("TTS interrupted, stopping audio stream")
                            break
                        if not line.strip():
                            continue
                        chunk_json = json.loads(line)
                        chunk_b64 = chunk_json.get("audio")
                        if not chunk_b64:
                            continue
                        chunk_bytes = base64.b64decode(chunk_b64)
                        await websocket.send_bytes(chunk_bytes)
        except Exception as e:
            print("TTS error:", e)

    async def process_user_utterance(audio_bytes):
        nonlocal conversation_history, tts_task, interrupt_event
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f2:
                wf2 = wave.open(f2, 'wb')
                wf2.setnchannels(1)
                wf2.setsampwidth(bytes_per_sample)
                wf2.setframerate(sample_rate)
                wf2.writeframes(audio_bytes)
                wf2.close()
                full_wav_path = f2.name

            await send_status("Transcribing...")
            try:
                result = model.transcribe(full_wav_path)
                text = result["text"].strip()
                print(f"Transcription: '{text}'")
            except Exception as e:
                print("Transcription error:", e)
                text = ""

            os.unlink(full_wav_path)

            if not text:
                await send_status("Listening...")
                return

            await send_status("Thinking...")

            try:
                messages = []
                for turn in conversation_history:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})
                messages.append({"role": "user", "content": text})

                response = ollama_client.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=messages
                )
                reply = response.choices[0].message.content
                print(f"LLM reply: {reply}")
            except Exception as e:
                print("LLM error:", e)
                reply = "Error contacting LLM"

            conversation_history.append({"user": text, "assistant": reply})

            await send_status("Speaking...")

            interrupt_event.clear()

            try:
                await websocket.send_text(json.dumps({"type": "transcript", "text": text, "reply": reply}))
                tts_task = asyncio.create_task(stream_tts(reply))
            except Exception as e:
                print("Error starting TTS task:", e)

            await send_status("Listening...")

        except asyncio.CancelledError:
            print("Processing task cancelled")
            return

    print("WebSocket connection open")
    try:
        await send_status("Listening...")
        while True:
            data = await websocket.receive_bytes()
            buffer += data
            audio_buffer += data

            while len(buffer) >= chunk_size:
                chunk = buffer[:chunk_size]
                buffer = buffer[chunk_size:]

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    wf = wave.open(f, 'wb')
                    wf.setnchannels(1)
                    wf.setsampwidth(bytes_per_sample)
                    wf.setframerate(sample_rate)
                    wf.writeframes(chunk)
                    wf.close()
                    wav_path = f.name

                speech_regions = vad_pipeline(wav_path)
                os.unlink(wav_path)

                speech_detected = False
                for segment, _ in speech_regions.itertracks():
                    duration = segment.end - segment.start
                    if duration >= 0.1:
                        speech_detected = True
                        break

                now = time.time()

                if speech_detected:
                    speech_counter += 1
                    if speech_counter >= speech_confirm_threshold:
                        if not user_speaking:
                            print("User started speaking")
                        user_speaking = True
                        silence_start = None

                        if tts_task and not tts_task.done() and not interrupt_event.is_set():
                            interrupt_event.set()
                            await send_status("Interrupting speech, listening...")
                            try:
                                await websocket.send_text(json.dumps({"type": "stop_audio"}))
                            except:
                                pass

                        if processing_task and not processing_task.done():
                            processing_task.cancel()
                            try:
                                await processing_task
                            except:
                                pass
                else:
                    speech_counter = 0
                    if user_speaking:
                        if silence_start is None:
                            silence_start = now
                        elif now - silence_start > silence_threshold:
                            print("User stopped speaking, processing utterance")
                            user_speaking = False
                            silence_start = None

                            if tts_task and not tts_task.done():
                                try:
                                    await asyncio.wait_for(tts_task, timeout=2)
                                except asyncio.TimeoutError:
                                    print("TTS task did not finish in time, continuing")
                                except Exception as e:
                                    print("Error waiting for TTS task:", e)
                                interrupt_event.clear()

                            if processing_task and not processing_task.done():
                                processing_task.cancel()
                                try:
                                    await processing_task
                                except:
                                    pass

                            processing_task = asyncio.create_task(process_user_utterance(audio_buffer))
                            audio_buffer = b""

    except Exception as e:
        print("WebSocket error:", e)
        try:
            await websocket.send_text(json.dumps({"type": "status", "message": "Error"}))
        except:
            pass