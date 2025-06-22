# Panopto Audio Processing Toolkit

A Python toolkit for downloading audio from Panopto lectures and transcribing them using either local Whisper models or OpenAI's API.

## Overview

This project provides tools to:
- Download audio from Panopto lecture recordings using `yt-dlp`
- Transcribe audio locally using Norwegian Whisper models
- Transcribe audio using OpenAI's Whisper API
- Process multiple lectures in batch

## Features

- **Audio Download**: Extract high-quality audio from Panopto lecture URLs
- **Local Transcription**: Use NB-Whisper models for Norwegian language transcription
- **API Transcription**: Leverage OpenAI's Whisper API for transcription
- **Batch Processing**: Handle multiple lectures at once
- **Authentication Support**: Uses cookies for accessing restricted content

## Prerequisites

### System Requirements
- Python 3.8+
- `yt-dlp` command-line tool
- Internet connection for downloading content

### External Dependencies
- **yt-dlp**: Install via `pip install yt-dlp` or from [GitHub](https://github.com/yt-dlp/yt-dlp)
- **FFmpeg**: Required by yt-dlp for audio processing

### Authentication
- `cookies.txt` file containing Panopto authentication cookies
- Place this file in the project root directory

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd panopto
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install yt-dlp:
```bash
pip install yt-dlp
```

## Usage

### 1. Audio Download

The `download_audio.py` script downloads audio from Panopto lectures:

```python
# Edit the script to specify your URLs and output folder
urls = ["https://ntnu.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=YOUR_ID"]
output_folder = "/path/to/your/audio/folder/"
download_audio(urls)
```

**Configuration:**
- Update `output_folder` to your desired save location
- Modify `format_id` based on `yt-dlp -F` output for optimal quality
- Adjust `audio_quality` as needed (default: 64K)

### 2. URL Management

Store lecture URLs in text files:
- `urls/objekt_urls.txt` - Object-oriented programming course URLs
- `urls/styring_urls.txt` - Control systems course URLs

Format:
```
Course Name - Date - Session
https://ntnu.cloud.panopto.eu/Panopto/Pages/Viewer.aspx?id=LECTURE_ID
```

### 3. Local Transcription

Use `transcribe_locally.py` for offline transcription with Norwegian Whisper:

```python
# Configure input and output paths
IN_PATH = "/path/to/audio/file.mp3"
OUT_PATH = "/path/to/output/text/file.txt"
```

**Features:**
- Uses NB-Whisper Small model for Norwegian language
- Chunk-based processing for large files
- Configurable beam search

### 4. API Transcription

Use `transcribe_with_api.py` for OpenAI Whisper API transcription:

```python
# Set your OpenAI API key
API_KEY = "your-openai-api-key"

# Configure audio file paths
audio_paths = ["/path/to/audio/file.mp3"]
```

**Note:** Remove the hardcoded API key and use environment variables for security.

## File Structure

```
panopto/
├── download_audio.py          # Main audio download script
├── transcribe_locally.py      # Local Whisper transcription
├── transcribe_with_api.py     # OpenAI API transcription
├── cookies.txt               # Panopto authentication cookies
├── urls/                     # URL collections
│   ├── objekt_urls.txt      # TDT4100 course URLs
│   └── styring_urls.txt     # TIØ4105 course URLs
├── misc/                     # Experimental and planning files
│   ├── experimental.ipynb   # Jupyter notebook for testing
│   ├── plan.txt            # Development roadmap
│   ├── prompt.txt          # AI prompts
│   └── test_audio.mp3      # Sample audio file
└── requirements.txt         # Python dependencies
```

## Configuration

### Audio Download Settings
- **Format ID**: Video+audio format selection (use `yt-dlp -F <url>` to see options)
- **Audio Quality**: Bitrate setting (default: 64K)
- **Output Format**: MP3 conversion

### Transcription Settings
- **Local Model**: NB-Whisper Small (Norwegian optimized)
- **Chunk Length**: 28 seconds for processing
- **Language**: Norwegian ("no")
- **Beam Search**: 5 beams for accuracy