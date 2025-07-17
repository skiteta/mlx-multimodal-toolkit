"""
MLX Multimodal Toolkit

A unified toolkit for working with Large Language Models (LLMs),
Vision Language Models (VLMs), and Audio Language Models (ALMs)
using the MLX framework on Apple Silicon.
"""

from .unified_driver import UnifiedModelDriver
from .models import LLMs, VLMs, ALMs
from .config import Config

__version__ = "0.1.0"
__all__ = ["UnifiedModelDriver", "LLMs", "VLMs", "ALMs", "Config"]