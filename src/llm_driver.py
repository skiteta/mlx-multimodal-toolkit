from mlx_lm import load, generate, stream_generate
from typing import Dict, Any, Optional, List, Generator
from .base_driver import BaseModelDriver
from .models import LLMs
import logging


class LLMDriver(BaseModelDriver):
    """Driver for Large Language Models"""
    
    def __init__(self, model_name: LLMs, config: Optional[Dict[str, Any]] = None):
        super().__init__(model_name.value, config)
        self.model_enum = model_name
        
    def load(self, adapter_path: Optional[str] = None, **kwargs):
        """Load the LLM model and tokenizer"""
        try:
            model_path = f"mlx-community/{self.model_name}"
            self._model, self._tokenizer = load(
                model_path, 
                adapter_path=adapter_path,
                **kwargs
            )
            self.logger.info(f"Successfully loaded LLM: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load LLM: {e}")
            raise
            
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response for a single prompt"""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        prompt = self.validate_input(prompt)
        
        # Apply chat template if not raw mode
        if not kwargs.get("raw", False):
            messages = [{"role": "user", "content": prompt}]
            if system_prompt := kwargs.get("system_prompt"):
                messages.insert(0, {"role": "system", "content": system_prompt})
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
        
        # Merge config with kwargs
        gen_kwargs = {
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.95,
            "repetition_penalty": 1.0,
        }
        gen_kwargs.update(self.config)
        gen_kwargs.update(kwargs)
        
        # Generate response
        response = generate(
            self._model, 
            self._tokenizer, 
            prompt=prompt,
            verbose=gen_kwargs.pop("verbose", False),
            **gen_kwargs
        )
        
        return response
        
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream tokens as they're generated"""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        prompt = self.validate_input(prompt)
        
        # Apply chat template if not raw mode
        if not kwargs.get("raw", False):
            messages = [{"role": "user", "content": prompt}]
            if system_prompt := kwargs.get("system_prompt"):
                messages.insert(0, {"role": "system", "content": system_prompt})
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
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
            self._tokenizer,
            prompt=prompt,
            **gen_kwargs
        ):
            yield token


def main():
    """Example usage of LLMDriver"""
    from models import LLMs
    
    # Initialize driver
    driver = LLMDriver(LLMs.PHI_3_5_MINI_INSTRUCT_4BIT)
    driver.load()
    
    # Generate response
    prompt = "日本の首都は？"
    response = driver.generate(prompt)
    print(f"Response: {response}")
    
    # Stream response
    print("\nStreaming response:")
    for token in driver.stream("AIについて短く説明してください"):
        print(token, end='', flush=True)
    print()


if __name__ == "__main__":
    main()
