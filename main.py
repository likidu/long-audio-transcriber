import requests
import json
import time
from pathlib import Path
import ffmpeg
import os
import math
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("WHISPER_API_KEY")

AUDIO_PATH = os.getenv("AUDIO_PATH", "/Users/ziberna/output_fast.wav")
OUTPUT_DIR = "output"
OUTPUT_PATH_SRT = os.path.join(OUTPUT_DIR, "transcription.srt")
CHUNK_DIR = os.path.join(OUTPUT_DIR, "temp_chunks")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "transcription_progress.json")
MAX_SIZE_MB = float(os.getenv("MAX_SIZE_MB", "24"))

def load_progress():
    """Load progress from file if it exists."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'processed_chunks': {}, 'completed': False}

def save_progress(chunk_path, transcription):
    """Save progress after each chunk is processed."""
    progress = load_progress()
    progress['processed_chunks'][chunk_path] = transcription
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def mark_completed():
    """Mark the transcription as completed."""
    progress = load_progress()
    progress['completed'] = True
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def setup_temp_directory():
    """Create temporary directory for chunks if it doesn't exist. If it exists, delete its contents."""
    if os.path.exists(CHUNK_DIR):
        for file in os.listdir(CHUNK_DIR):
            os.remove(os.path.join(CHUNK_DIR, file))
    else:
        os.makedirs(CHUNK_DIR)

def cleanup_temp_directory():
    """Remove temporary chunks."""
    if os.path.exists(CHUNK_DIR):
        for file in os.listdir(CHUNK_DIR):
            os.remove(os.path.join(CHUNK_DIR, file))
        os.rmdir(CHUNK_DIR)

def get_audio_duration(file_path):
    """Get duration of audio file in seconds."""
    probe = ffmpeg.probe(file_path)
    audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    return float(probe['format']['duration'])

def split_audio_file(file_path):
    """Split audio file into chunks smaller than 25MB."""
    print(f"Loading audio file: {file_path}")
    
    total_duration = get_audio_duration(file_path)
    file_size = os.path.getsize(file_path)
    
    # Calculate how many chunks we need
    num_chunks = math.ceil(file_size / (MAX_SIZE_MB * 1024 * 1024))
    chunk_duration = total_duration / num_chunks

    chunks = []
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = os.path.join(CHUNK_DIR, f"chunk_{i:03d}.mp3")

        # Skip if chunk already exists and is processed
        progress = load_progress()
        if chunk_path in progress['processed_chunks']:
            print(f"Chunk {i+1} already processed, skipping creation")
            chunks.append(chunk_path)
            continue

        print(f"Creating chunk {i+1}: {chunk_path}")

        # Use ffmpeg to extract chunk as mp3 to keep file size small
        stream = ffmpeg.input(file_path, ss=start_time, t=chunk_duration)
        stream = ffmpeg.output(stream, chunk_path, acodec='libmp3lame', audio_bitrate='64k')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        chunks.append(chunk_path)
    
    return chunks

def transcribe_chunk(chunk_path, is_first_chunk=""):
    """Transcribe a single chunk with timestamps."""
    print(f"\nProcessing chunk: {chunk_path}")
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    max_retries = 5
    for attempt in range(max_retries):
        with open(chunk_path, "rb") as audio_file:
            files = {"file": audio_file}
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment"
            }

            if is_first_chunk:
                print("Sending request for first chunk...")
            response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
            print(f"Rate limited (429). Retrying in {retry_after}s (attempt {attempt + 1}/{max_retries})...")
            print(f"Response: {response.text}")
            time.sleep(retry_after)
            continue

        if response.status_code >= 500:
            wait_time = 2 ** attempt
            print(f"Server error ({response.status_code}). Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)
            continue

        if response.status_code == 400:
            print(f"Bad request (400). Response: {response.text}")
        response.raise_for_status()
        return response.json()

    response.raise_for_status()  # raise after all retries exhausted

def format_srt_time(seconds):
    """Format seconds into SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def merge_transcriptions(transcriptions, chunk_durations):
    """Merge verbose_json transcription chunks into SRT, adjusting timestamps."""
    all_segments = []
    time_offset = 0

    for i, result in enumerate(transcriptions):
        for seg in result.get('segments', []):
            start = seg['start'] + time_offset
            end = seg['end'] + time_offset
            text = seg['text'].strip()
            all_segments.append((start, end, text))
        time_offset += chunk_durations[i]

    srt_lines = []
    for idx, (start, end, text) in enumerate(all_segments, 1):
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        srt_lines.append(text)
        srt_lines.append("")

    return '\n'.join(srt_lines)

def main():
    try:
        # Check if previous transcription was completed
        progress = load_progress()
        if progress['completed']:
            print("Previous transcription was completed. Remove progress file to start over.")
            return
            
        print("Starting transcription process...")
        
        # Check if file exists
        if not Path(AUDIO_PATH).is_file():
            print(f"Error: Audio file not found at {AUDIO_PATH}")
            return
        
        # Create output and temporary directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        setup_temp_directory()
        
        # Split file if needed
        file_size_mb = os.path.getsize(AUDIO_PATH) / (1024 * 1024)
        if file_size_mb > MAX_SIZE_MB:
            print(f"File size ({file_size_mb:.1f}MB) exceeds {MAX_SIZE_MB}MB limit. Splitting file...")
            chunk_paths = split_audio_file(AUDIO_PATH)
        else:
            chunk_paths = [AUDIO_PATH]
        
        # Get chunk durations for timestamp offset calculation
        chunk_durations = []
        for chunk_path in chunk_paths:
            chunk_durations.append(get_audio_duration(chunk_path))

        # Process each chunk
        transcriptions = []
        for i, chunk_path in enumerate(chunk_paths):
            try:
                # Check if chunk was already processed
                progress = load_progress()
                if chunk_path in progress['processed_chunks']:
                    print(f"Loading previously processed chunk {i+1}/{len(chunk_paths)}")
                    transcriptions.append(progress['processed_chunks'][chunk_path])
                    continue

                is_first_chunk = (i == 0)
                result = transcribe_chunk(chunk_path, is_first_chunk)
                transcriptions.append(result)

                # Save progress after each chunk
                save_progress(chunk_path, result)

                print(f"Successfully transcribed chunk {i+1}/{len(chunk_paths)}")
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                raise

        # Merge results
        print("\nMerging transcriptions...")
        merged_result = merge_transcriptions(transcriptions, chunk_durations)

        # Save results
        print(f"Saving SRT transcription to {OUTPUT_PATH_SRT}")
        with open(OUTPUT_PATH_SRT, "w", encoding="utf-8") as f:
            f.write(merged_result)

        # Mark transcription as completed
        mark_completed()

        # Print sample
        print("\nTranscription completed successfully!")
        print(f"SRT saved to: {OUTPUT_PATH_SRT}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        # Don't cleanup if there was an error - keep chunks for resuming
        if progress.get('completed', False):
            cleanup_temp_directory()

if __name__ == "__main__":
    main()