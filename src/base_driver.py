from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator
import logging


class BaseModelDriver(ABC):
    """Base class for all model drivers"""
    
    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = None
        self._tokenizer = None
        self._processor = None
        
    @abstractmethod
    def load(self, **kwargs):
        """Load the model and necessary components"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response for a single prompt"""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream tokens as they're generated"""
        pass
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Process multiple prompts"""
        return [self.generate(prompt, **kwargs) for prompt in prompts]
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "config": self.config,
            "loaded": self._model is not None
        }
    
    def validate_input(self, prompt: Any) -> str:
        """Validate and convert input to string"""
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        return str(prompt)