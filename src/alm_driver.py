import mlx.core as mx
from typing import Dict, Any, Optional, Generator
from .base_driver import BaseModelDriver
from .models import ALMs
import logging


class ALMDriver(BaseModelDriver):
    """Driver for Audio Language Models"""
    
    def __init__(self, model_name: ALMs, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name.value, config)
        self.model_enum = model_name
        
    def load(self, **kwargs):
        """Load the ALM model"""
        try:
            model_path = f"mlx-community/{self.model_name}"
            
            if "whisper" in self.model_name.lower():
                from mlx_whisper import load_model
                self._model = load_model(model_path, **kwargs)
            elif "kokoro" in self.model_name.lower():
                # Placeholder for Kokoro TTS model loading
                raise NotImplementedError("Kokoro TTS model support coming soon")
            else:
                raise ValueError(f"Unknown ALM model type: {self.model_name}")
                
            self.logger.info(f"Successfully loaded ALM model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ALM model: {e}")
            raise
            
    def generate(self, audio_file: str = None, prompt: str = None, **kwargs) -> str:
        """Generate transcription or audio from the model"""
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        if "whisper" in self.model_name.lower():
            if not audio_file:
                raise ValueError("Audio file path required for transcription")
            return self._transcribe_audio(audio_file, **kwargs)
        elif "kokoro" in self.model_name.lower():
            if not prompt:
                raise ValueError("Text prompt required for TTS")
            return self._generate_audio(prompt, **kwargs)
            
    def stream(self, audio_file: str = None, prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """Stream transcription tokens"""
        if "whisper" in self.model_name.lower():
            # Whisper doesn't support streaming by default
            result = self.generate(audio_file=audio_file, **kwargs)
            yield result
        else:
            raise NotImplementedError("Streaming not supported for this model")
            
    def _transcribe_audio(self, audio_file: str, **kwargs) -> str:
        """Transcribe audio using Whisper"""
        from mlx_whisper import transcribe
        
        options = {
            "path_or_hf_repo": f"mlx-community/{self.model_name}",
            "audio": audio_file,
            "verbose": kwargs.get("verbose", True),
            "language": kwargs.get("language", None),
            "temperature": kwargs.get("temperature", 0),
            "word_timestamps": kwargs.get("word_timestamps", False),
        }
        
        result = transcribe(**options)
        return result["text"]
        
    def _generate_audio(self, text: str, **kwargs) -> str:
        """Generate audio using TTS model"""
        raise NotImplementedError("TTS generation coming soon")