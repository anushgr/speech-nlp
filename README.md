# 🎙️ Whisper Audio Transcriber

This is a simple Python project that uses [OpenAI's Whisper](https://github.com/openai/whisper) model to transcribe audio files (.wav, .mp3, .m4a, .flac) into text. Ideal for quick voice-to-text processing on local machines.

---

## 🔧 Features

- ✅ Automatic speech recognition using Whisper
- 🔍 Optional language detection (or specify manually)
- 🛠️ Validates audio file format and accessibility
- 📁 Outputs transcription to a `.txt` file
- 🔗 FFmpeg integration for format compatibility

---

## 🖥️ Requirements

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) (must be installed and in your system PATH)

---

## 📦 Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/whisper-audio-transcriber.git
   cd whisper-audio-transcriber
