#!/usr/bin/env python3
"""
Examples for audio processing using Audio Language Models (ALMs)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_driver import UnifiedModelDriver
from src.models import ALMs


def basic_transcription():
    """Basic audio transcription example"""
    print("=== Basic Audio Transcription ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)
    
    audio_file = "sample_audio.wav"
    
    try:
        transcription = model.generate(audio_file=audio_file)
        print(f"Audio file: {audio_file}")
        print(f"Transcription: {transcription}")
        print()
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        print("Make sure the audio file exists and is in a supported format.")
        print()


def multilingual_transcription():
    """Multilingual transcription example"""
    print("=== Multilingual Transcription ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)
    
    audio_files = [
        ("english_speech.wav", "en"),
        ("japanese_speech.wav", "ja"),
        ("french_speech.wav", "fr"),
        ("spanish_speech.wav", "es")
    ]
    
    for audio_file, language in audio_files:
        try:
            transcription = model.generate(
                audio_file=audio_file,
                language=language,
                temperature=0.0  # More deterministic for transcription
            )
            print(f"Audio: {audio_file} (Language: {language})")
            print(f"Transcription: {transcription}")
            print()
        except Exception as e:
            print(f"Error transcribing {audio_file}: {e}")
            print()


def batch_transcription():
    """Batch transcription of multiple audio files"""
    print("=== Batch Transcription ===")
    
    driver = UnifiedModelDriver()
    
    # Prepare batch inputs
    inputs = [
        {"audio_file": "interview1.wav"},
        {"audio_file": "meeting_recording.wav"},
        {"audio_file": "lecture_part1.wav"},
        {"audio_file": "podcast_segment.wav"}
    ]
    
    try:
        transcriptions = driver.batch_process(inputs, ALMs.WHISPER_LARGE_V3_MLX)
        
        for i, (input_data, transcription) in enumerate(zip(inputs, transcriptions), 1):
            print(f"File {i}: {input_data['audio_file']}")
            print(f"Transcription: {transcription[:100]}...")
            print()
            
    except Exception as e:
        print(f"Error in batch transcription: {e}")
        print()


def transcription_with_timestamps():
    """Transcription with word-level timestamps"""
    print("=== Transcription with Timestamps ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)
    
    audio_file = "timestamped_audio.wav"
    
    try:
        # Enable word timestamps
        transcription = model.generate(
            audio_file=audio_file,
            word_timestamps=True,
            verbose=True
        )
        print(f"Audio file: {audio_file}")
        print(f"Transcription with timestamps: {transcription}")
        print()
    except Exception as e:
        print(f"Error generating timestamped transcription: {e}")
        print()


def streaming_transcription():
    """Streaming transcription (simulated)"""
    print("=== Streaming Transcription ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)
    
    audio_file = "long_audio.wav"
    
    print(f"Audio file: {audio_file}")
    print("Transcription (streaming): ", end="")
    
    try:
        for token in model.stream(audio_file=audio_file):
            print(token, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"Error streaming transcription: {e}")
        print()


def custom_transcription_settings():
    """Transcription with custom settings"""
    print("=== Custom Transcription Settings ===")
    
    driver = UnifiedModelDriver()
    
    # Custom configuration for transcription
    config = {
        "temperature": 0.0,  # Deterministic output
        "language": "ja",    # Japanese language
        "verbose": False     # Minimal output
    }
    
    model = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX, config=config)
    
    audio_file = "japanese_audio.wav"
    
    try:
        transcription = model.generate(audio_file=audio_file)
        print(f"Audio file: {audio_file}")
        print(f"Japanese transcription: {transcription}")
        print()
    except Exception as e:
        print(f"Error with custom settings: {e}")
        print()


def audio_analysis():
    """Audio content analysis"""
    print("=== Audio Content Analysis ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)
    
    audio_file = "conversation.wav"
    
    try:
        # First, get transcription
        transcription = model.generate(audio_file=audio_file)
        
        print(f"Audio file: {audio_file}")
        print(f"Transcription: {transcription}")
        
        # Analyze the content (this would require an LLM for analysis)
        print("\nNote: For content analysis, you would typically:")
        print("1. Transcribe the audio using ALM")
        print("2. Process the transcription with an LLM for analysis")
        print("3. Examples: sentiment analysis, topic extraction, summarization")
        print()
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        print()


def tts_example():
    """Text-to-Speech example (placeholder)"""
    print("=== Text-to-Speech Example ===")
    
    # Note: This is a placeholder since Kokoro TTS isn't fully implemented
    print("Text-to-Speech with Kokoro model:")
    print("This feature is coming soon!")
    print()
    
    print("Example usage would be:")
    print("driver = UnifiedModelDriver()")
    print("model = driver.load_model(ALMs.KOKORO_82M_BF16)")
    print("audio_output = model.generate(prompt='Hello, world!')")
    print()


def main():
    """Run all audio processing examples"""
    print("MLX Multimodal Toolkit - Audio Processing Examples")
    print("=" * 50)
    print()
    
    print("Note: Make sure you have sample audio files in the current directory")
    print("or update the audio file paths in the examples.")
    print("Supported formats: WAV, MP3, FLAC, M4A, etc.")
    print()
    
    try:
        basic_transcription()
        multilingual_transcription()
        batch_transcription()
        transcription_with_timestamps()
        streaming_transcription()
        custom_transcription_settings()
        audio_analysis()
        tts_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())