FROM python:3.9-slim

# Install ffmpeg and build tools for webrtcvad
RUN apt-get update && apt-get install -y ffmpeg build-essential

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY static/ ./static/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8010"]