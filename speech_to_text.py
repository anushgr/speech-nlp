import whisper
import os
from pathlib import Path
import subprocess
import shutil

# Hardcoded audio file path
AUDIO_PATH = "./test.wav"
# Hardcoded language (set to None for auto-detection, or e.g., "en" for English)
LANGUAGE = None

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print(f"FFmpeg version: {result.stdout.splitlines()[0]}")
        return True
    except FileNotFoundError:
        return False

def transcribe_audio(audio_path, model_name="small", language=None):
    """
    Transcribe an audio file using the Whisper Small model.
    
    Args:
        audio_path (str): Path to the audio file (e.g., .wav, .mp3).
        model_name (str): Whisper model to use (default: 'small').
        language (str): Language code (e.g., 'en' for English, None for auto-detection).
    
    Returns:
        str: Transcribed text or error message.
    """
    try:
        # Check ffmpeg availability
        if not check_ffmpeg():
            return "Error: FFmpeg is not installed or not in system PATH. Install FFmpeg and add it to PATH."

        # Validate audio file
        audio_path = str(Path(audio_path))  # Normalize path
        print(f"Checking file: {audio_path}")
        if not os.path.isfile(audio_path):
            return f"Error: '{audio_path}' is not a valid file."
        if not any(audio_path.lower().endswith(ext) for ext in ['.wav', '.mp3', '.m4a', '.flac']):
            return f"Error: '{audio_path}' is not a supported audio format (use .wav, .mp3, .m4a, or .flac)."

        # Check file accessibility
        try:
            with open(audio_path, 'rb') as f:
                print(f"File '{audio_path}' is accessible.")
        except Exception as e:
            return f"Error: Cannot access file '{audio_path}': {str(e)}"

        # Load the Whisper model
        print(f"Loading Whisper '{model_name}' model...")
        model = whisper.load_model(model_name)
        
        # Transcribe the audio
        print(f"Transcribing '{audio_path}'...")
        options = {"language": language} if language else {}
        result = model.transcribe(audio_path, **options)
        
        # Extract the transcribed text
        transcribed_text = result["text"].strip()
        if not transcribed_text:
            return "Warning: No speech detected in the audio."
        
        # Detect language if not specified
        detected_language = result.get("language", "unknown")
        print(f"Detected language: {detected_language}")
        
        return transcribed_text
    
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def main():
    # Run transcription with hardcoded audio path and language
    transcription = transcribe_audio(AUDIO_PATH, model_name="small", language=LANGUAGE)
    
    # Output the result
    print("\nTranscription:")
    print(transcription)
    
    # Save to a file
    output_file = "transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"\nTranscription saved to '{output_file}'")

if __name__ == "__main__":
    main()

