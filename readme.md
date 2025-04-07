# We have conversational AI at home. 
#### Self-hosted conversational voice chat

## Features
- Real-time streaming voice input with VAD-based segmentation
- Transcription and LLM response generation
- Streaming TTS playback with interruption support
- Simple web interface for interaction

This project provides a real-time voice chat interface powered by:
- **Voice Activity Detection (VAD)** using `pyannote.audio`
- **Speech recognition** via OpenAI's Whisper
- **Conversational AI** using an Ollama-hosted LLM
- **Text-to-speech (TTS)** via Kokoro

---

## Quickstart

```bash
# Download the docker-compose file
curl -O https://raw.githubusercontent.com/gbstox/convo_ai_at_home/main/docker-compose.yml

# Create a .env file with your Hugging Face token
echo "HF_TOKEN=your_huggingface_token_here" > .env

# (Optional) Set a specific Ollama model
echo "OLLAMA_MODEL=gemma3:12b" >> .env

# Start everything
docker compose up --build
```

Then open your browser at: [http://localhost:8010](http://localhost:8010)

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/gbstox/convo_ai_at_home.git
cd convo_ai_at_home
```

### 2. Configure Hugging Face Token

This app requires a Hugging Face token to download the `pyannote/segmentation-3.0` model for VAD.

- Obtain a token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Create a `.env` file in the root directory (if not already present):
```
HF_TOKEN=your_huggingface_token_here
OLLAMA_MODEL=gemma3:12b # optional, override default model
```


**Note:** Your `.env` file is already included in `.gitignore` (recommended) to avoid leaking secrets.

### 3. Build and start the services

Make sure you have Docker and Docker Compose installed.

```bash
docker compose up --build
```

This will:
- Pull and start **Ollama** (for LLM)
- Pull and start **Kokoro** (for TTS)
- Build and start the **backend** FastAPI app

---

## Usage

Once all containers are running, open your browser and navigate to: [http://localhost:8010](http://localhost:8010)

You will see a simple web interface with a **Start Conversation** button.

- Click **Start Conversation** to begin.
- Speak into your microphone.
- The app will detect speech, transcribe it, generate a response, and play it back.
- The conversation history will be displayed on the page.

---

## Environment Variables

- `HF_TOKEN` (required): Your Hugging Face API token.
- `OLLAMA_MODEL` (optional): The Ollama model to use (default: `gemma3:12b`). You can override this in `.env` or `docker-compose.yml`.

---

## Notes

- The backend listens on port **8010**.
- Ollama listens on port **11434**.
- Kokoro listens on port **8880**.
- The app will automatically pull the specified Ollama model if not present.
- The TTS voice is set to `af_bella` by default.

---

## Troubleshooting

- **Model download issues:** Ensure your Hugging Face token has access to `pyannote/segmentation-3.0`.
- **Microphone access:** Allow your browser to access the microphone.
- **Performance:** Running large models may require significant resources.

---

## License

MIT or your preferred license.

---

## Acknowledgments

- [Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Ollama](https://ollama.com/)
- [Kokoro TTS](https://github.com/remsky/kokoro)