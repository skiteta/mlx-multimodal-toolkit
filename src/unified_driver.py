from typing import Union, Optional, Dict, Any, List
from pathlib import Path
import logging
import os

from .models import LLMs, VLMs, ALMs
from .llm_driver import LLMDriver
from .vlm_driver import VLMDriver
from .alm_driver import ALMDriver
from .base_driver import BaseModelDriver


class UnifiedModelDriver:
    """Unified interface for all model types"""
    
    def __init__(self, cache_dir: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize the unified driver
        
        Args:
            cache_dir: Directory to cache models (default: ~/.cache/mlx_models)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/mlx_models")
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Model registry
        self._models: Dict[str, BaseModelDriver] = {}
        self._active_model: Optional[BaseModelDriver] = None
        
    def load_model(
        self, 
        model_enum: Union[LLMs, VLMs, ALMs],
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseModelDriver:
        """
        Load a model of any type
        
        Args:
            model_enum: Model enum from LLMs, VLMs, or ALMs
            config: Model configuration
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model driver
        """
        # Determine model type and create appropriate driver
        if isinstance(model_enum, LLMs):
            driver = LLMDriver(model_enum, config)
        elif isinstance(model_enum, VLMs):
            driver = VLMDriver(model_enum, config)
        elif isinstance(model_enum, ALMs):
            driver = ALMDriver(model_enum, config)
        else:
            raise ValueError(f"Unknown model type: {type(model_enum)}")
            
        # Load the model
        try:
            self.logger.info(f"Loading model: {model_enum.value}")
            driver.load(**kwargs)
            
            # Store in registry
            model_key = model_enum.value
            self._models[model_key] = driver
            self._active_model = driver
            
            self.logger.info(f"Successfully loaded: {model_enum.value}")
            return driver
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_enum.value}: {e}")
            raise
            
    def get_model(self, model_name: str) -> Optional[BaseModelDriver]:
        """Get a loaded model by name"""
        return self._models.get(model_name)
        
    def list_loaded_models(self) -> List[str]:
        """List all loaded models"""
        return list(self._models.keys())
        
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        if model_name in self._models:
            del self._models[model_name]
            if self._active_model and self._active_model.model_name == model_name:
                self._active_model = None
            self.logger.info(f"Unloaded model: {model_name}")
            return True
        return False
        
    def get_active_model(self) -> Optional[BaseModelDriver]:
        """Get the currently active model"""
        return self._active_model
        
    def set_active_model(self, model_name: str) -> bool:
        """Set the active model"""
        if model_name in self._models:
            self._active_model = self._models[model_name]
            return True
        return False
        
    def compare_models(
        self, 
        models: List[Union[LLMs, VLMs, ALMs]], 
        prompt: str,
        **kwargs
    ) -> Dict[str, str]:
        """
        Compare outputs from multiple models
        
        Args:
            models: List of model enums to compare
            prompt: Input prompt
            **kwargs: Additional generation arguments
            
        Returns:
            Dictionary mapping model names to their outputs
        """
        results = {}
        
        for model_enum in models:
            try:
                # Load model if not already loaded
                model_name = model_enum.value
                if model_name not in self._models:
                    self.load_model(model_enum)
                    
                driver = self._models[model_name]
                
                # Generate response based on model type
                if isinstance(driver, VLMDriver):
                    response = driver.generate(prompt, images=kwargs.get("images"), **kwargs)
                elif isinstance(driver, ALMDriver):
                    response = driver.generate(
                        audio_file=kwargs.get("audio_file"),
                        prompt=prompt if "kokoro" in model_name.lower() else None,
                        **kwargs
                    )
                else:
                    response = driver.generate(prompt, **kwargs)
                    
                results[model_name] = response
                
            except Exception as e:
                self.logger.error(f"Error with model {model_name}: {e}")
                results[model_name] = f"Error: {str(e)}"
                
        return results
        
    def batch_process(
        self,
        inputs: List[Dict[str, Any]],
        model_enum: Union[LLMs, VLMs, ALMs],
        **kwargs
    ) -> List[str]:
        """
        Process multiple inputs with the same model
        
        Args:
            inputs: List of input dictionaries with 'prompt' and optional 'images'/'audio_file'
            model_enum: Model to use
            **kwargs: Additional generation arguments
            
        Returns:
            List of responses
        """
        # Load model if needed
        model_name = model_enum.value
        if model_name not in self._models:
            self.load_model(model_enum)
            
        driver = self._models[model_name]
        results = []
        
        for input_data in inputs:
            try:
                prompt = input_data.get("prompt", "")
                
                if isinstance(driver, VLMDriver):
                    response = driver.generate(
                        prompt, 
                        images=input_data.get("images"),
                        **kwargs
                    )
                elif isinstance(driver, ALMDriver):
                    response = driver.generate(
                        audio_file=input_data.get("audio_file"),
                        prompt=prompt if "kokoro" in model_name.lower() else None,
                        **kwargs
                    )
                else:
                    response = driver.generate(prompt, **kwargs)
                    
                results.append(response)
                
            except Exception as e:
                self.logger.error(f"Error processing input: {e}")
                results.append(f"Error: {str(e)}")
                
        return results