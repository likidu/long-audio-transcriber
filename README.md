# Long Audio Transcriber

Transcribe long audio files to SRT subtitles using OpenAI's Whisper API. Large files are automatically split into chunks, transcribed individually, and merged back with correct timestamps.

## Features

- Automatic chunking for files exceeding the API size limit (default 24 MB)
- Chunks are re-encoded to MP3 at 64 kbps to keep upload size small
- Outputs standard SRT subtitle format with segment-level timestamps
- Progress tracking — resume interrupted transcriptions where they left off
- Retry with exponential backoff on rate limits (429) and server errors (5xx)

## Prerequisites

- Python 3.8+
- ffmpeg
- OpenAI API key

### Installing ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/likidu/long-audio-transcriber.git
cd long-audio-transcriber
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```
WHISPER_API_KEY="your-openai-api-key"
AUDIO_PATH="/path/to/your/audio.mp3"
```

## Usage

```bash
python main.py
```

The tool will:

1. Check if the file exceeds the size limit and split it into MP3 chunks if needed
2. Send each chunk to the Whisper API for transcription
3. Merge all chunks into a single SRT file with corrected timestamps
4. Save the result to `output/transcription.srt`

## Output

All output is written to the `output/` directory:

| File | Description |
|------|-------------|
| `transcription.srt` | Final SRT subtitle file |
| `transcription_progress.json` | Progress tracker for resuming |
| `temp_chunks/` | Temporary MP3 chunks (cleaned up on completion) |

## Configuration

Environment variables (set in `.env` or shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_API_KEY` | — | OpenAI API key (required) |
| `AUDIO_PATH` | — | Path to the input audio file |
| `MAX_SIZE_MB` | `24` | Max chunk size in MB before splitting |

## Error Handling

- Progress is saved after each chunk, so a crash or rate limit won't lose work
- On restart, already-processed chunks are skipped automatically
- Delete `output/transcription_progress.json` to start a fresh transcription
