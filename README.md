# MLX Multimodal Toolkit

A unified toolkit for working with Large Language Models (LLMs), Vision Language Models (VLMs), and Audio Language Models (ALMs) using the MLX framework on Apple Silicon.

## Features

- **Unified Interface**: Single API for all model types (LLM, VLM, ALM)
- **Model Support**: 
  - LLMs: Phi-3.5, SmolLM, Qwen2.5, Mistral, Mixtral, Yi, OLMoE, Mamba-Codestral
  - VLMs: Florence-2, Qwen2.5-VL, olmOCR, Molmo, SmolVLM
  - ALMs: Kokoro, Whisper
- **Easy Configuration**: Simple model selection and configuration
- **Batch Processing**: Support for batch inference
- **Streaming Support**: Real-time text generation
- **Error Handling**: Comprehensive error handling and validation

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/mlx-multimodal-toolkit.git
cd mlx-multimodal-toolkit

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# For development dependencies
uv pip install -e ".[dev]"
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/mlx-multimodal-toolkit.git
cd mlx-multimodal-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.9+
- uv package manager (recommended) or pip

## Quick Start

### Using LLMs

```python
from src.unified_driver import UnifiedModelDriver
from src.models import LLMs

# Initialize the driver
driver = UnifiedModelDriver()

# Load an LLM
model = driver.load_model(LLMs.PHI_3_5_MINI_INSTRUCT_4BIT)

# Generate text
response = model.generate("What is the capital of Japan?")
print(response)

# Stream generation
for token in model.stream("Tell me a story about AI"):
    print(token, end='', flush=True)
```

### Using VLMs

```python
from src.unified_driver import UnifiedModelDriver
from src.models import VLMs

# Initialize and load VLM
driver = UnifiedModelDriver()
model = driver.load_model(VLMs.FLORENCE_2_LARGE_FT_BF16)

# Analyze an image
response = model.generate(
    "What's in this image?", 
    images=["path/to/image.jpg"]
)
print(response)
```

### Using ALMs

```python
from src.unified_driver import UnifiedModelDriver
from src.models import ALMs

# Initialize and load ALM
driver = UnifiedModelDriver()
model = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)

# Transcribe audio
transcription = model.generate(audio_file="path/to/audio.wav")
print(transcription)
```

## Advanced Usage

### Batch Processing

```python
# Process multiple prompts
prompts = [
    "Translate 'Hello' to Japanese",
    "What is 2 + 2?",
    "Explain quantum computing"
]

responses = model.batch_generate(prompts)
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}\nA: {response}\n")
```

### Custom Configuration

```python
# Load model with custom settings
config = {
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 0.9
}

model = driver.load_model(
    LLMs.QWEN2_5_32B_INSTRUCT_BF16, 
    config=config
)
```

### Model Comparison

```python
# Compare different models
models = [
    LLMs.PHI_3_5_MINI_INSTRUCT_4BIT,
    LLMs.SMOLLM_1_7B_FP16,
    LLMs.MISTRAL_NEMO_INSTRUCT_2407_BF16
]

prompt = "Explain the concept of recursion"

for model_enum in models:
    model = driver.load_model(model_enum)
    response = model.generate(prompt)
    print(f"\n{model_enum.value}:\n{response}")
```

## Examples

Check the `examples/` directory for more detailed examples:

- `text_generation.py` - Various LLM usage examples
- `image_analysis.py` - VLM examples for image understanding
- `audio_processing.py` - ALM examples for transcription
- `multimodal_pipeline.py` - Combined usage of different models

## API Reference

### UnifiedModelDriver

The main interface for loading and managing models.

```python
driver = UnifiedModelDriver(cache_dir="~/.cache/mlx_models")
```

### Model Methods

All models support these core methods:

- `generate(prompt, **kwargs)` - Generate response for a single prompt
- `stream(prompt, **kwargs)` - Stream tokens as they're generated
- `batch_generate(prompts, **kwargs)` - Process multiple prompts
- `get_info()` - Get model information and capabilities

## Configuration

Create a `config.yaml` file for persistent settings:

```yaml
models:
  default_llm: "Phi-3.5-mini-instruct-4bit"
  default_vlm: "Florence-2-large-ft-bf16"
  default_alm: "whisper-large-v3-mlx"

generation:
  max_tokens: 500
  temperature: 0.8
  top_p: 0.95

system:
  cache_dir: "~/.cache/mlx_models"
  logging_level: "INFO"
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Try using quantized models (4-bit or 8-bit versions)
2. **Slow Generation**: Ensure you're using Metal acceleration
3. **Model Loading Errors**: Check internet connection and disk space

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- MLX team at Apple for the excellent framework
- Model creators and the open-source AI community
- Contributors and users of this toolkit