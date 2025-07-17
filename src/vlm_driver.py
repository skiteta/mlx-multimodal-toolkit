from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import (
    load,
    load_config,
    generate,
    stream_generate,
)
from typing import Dict, Any, Optional, List, Union, Generator
from .base_driver import BaseModelDriver
from .models import VLMs
import logging
from pathlib import Path


class VLMDriver(BaseModelDriver):
    """Driver for Vision Language Models"""
    
    def __init__(self, model_name: VLMs, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name.value, config)
        self.model_enum = model_name
        self._config = None
        
    def load(self, adapter_path: Optional[str] = None, **kwargs):
        """Load the VLM model and processor"""
        try:
            model_path = f"mlx-community/{self.model_name}"
            self._config = load_config(model_path, trust_remote_code=True)
            self._model, self._processor = load(
                model_path, 
                adapter_path=adapter_path, 
                lazy=kwargs.get("lazy", False), 
                trust_remote_code=True
            )
            self.logger.info(f"Successfully loaded VLM: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load VLM: {e}")
            raise
            
    def generate(self, prompt: str, images: Optional[Union[str, List[str]]] = None, **kwargs) -> str:
        """Generate response for prompt with optional images"""
        if not self._model or not self._processor:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        prompt = self.validate_input(prompt)
        
        # Handle image input
        if images is None:
            images = []
        elif isinstance(images, str):
            images = [images]
        
        # Validate image paths
        validated_images = []
        for img in images:
            if isinstance(img, str) and Path(img).exists():
                validated_images.append(img)
            else:
                self.logger.warning(f"Image not found: {img}")
                
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        if system_prompt := kwargs.get("system_prompt"):
            messages.insert(0, {"role": "system", "content": system_prompt})
            
        formatted_prompt = apply_chat_template(
            self._processor, 
            self._config, 
            messages, 
            num_images=len(validated_images)
        )
        
        # Merge config with kwargs
        gen_kwargs = {
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.95,
        }
        gen_kwargs.update(self.config)
        gen_kwargs.update(kwargs)
        
        # Generate response
        response = generate(
            self._model, 
            self._processor, 
            formatted_prompt, 
            validated_images,
            verbose=gen_kwargs.pop("verbose", False),
            **gen_kwargs
        )
        
        return response
        
    def stream(self, prompt: str, images: Optional[Union[str, List[str]]] = None, **kwargs) -> Generator[str, None, None]:
        """Stream tokens as they're generated"""
        if not self._model or not self._processor:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        prompt = self.validate_input(prompt)
        
        # Handle image input
        if images is None:
            images = []
        elif isinstance(images, str):
            images = [images]
            
        # Validate image paths
        validated_images = []
        for img in images:
            if isinstance(img, str) and Path(img).exists():
                validated_images.append(img)
                
        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = apply_chat_template(
            self._processor, 
            self._config, 
            messages, 
            num_images=len(validated_images)
        )
        
        # Merge config with kwargs
        gen_kwargs = {
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.95,
        }
        gen_kwargs.update(self.config)
        gen_kwargs.update(kwargs)
        
        # Stream generate
        for token in stream_generate(
            self._model,
            self._processor,
            formatted_prompt,
            validated_images,
            **gen_kwargs
        ):
            yield token


def main():
    """Example usage of VLMDriver"""
    from models import VLMs
    
    # Initialize driver
    driver = VLMDriver(VLMs.SMOLVLM_INSTRUCT_BF16)
    driver.load()
    
    # Generate response with image
    prompt = "画像には何が写っていますか？詳しく説明してください。"
    response = driver.generate(prompt, images=["example.jpg"])
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
