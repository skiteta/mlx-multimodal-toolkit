#!/usr/bin/env python3
"""
Examples for text generation using Large Language Models (LLMs)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_driver import UnifiedModelDriver
from src.models import LLMs


def basic_text_generation():
    """Basic text generation example"""
    print("=== Basic Text Generation ===")
    
    # Initialize driver
    driver = UnifiedModelDriver()
    
    # Load a smaller model for quick testing
    model = driver.load_model(LLMs.PHI_3_5_MINI_INSTRUCT_4BIT)
    
    # Generate response
    prompt = "What is the capital of Japan?"
    response = model.generate(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print()


def streaming_generation():
    """Streaming text generation example"""
    print("=== Streaming Generation ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(LLMs.SMOLLM_1_7B_FP16)
    
    prompt = "Write a short story about artificial intelligence in the future."
    print(f"Prompt: {prompt}")
    print("Response (streaming): ", end="")
    
    for token in model.stream(prompt):
        print(token, end="", flush=True)
    
    print("\n")


def custom_configuration():
    """Using custom configuration for generation"""
    print("=== Custom Configuration ===")
    
    driver = UnifiedModelDriver()
    
    # Custom configuration
    config = {
        "max_tokens": 150,
        "temperature": 0.5,
        "top_p": 0.9
    }
    
    model = driver.load_model(LLMs.QWEN2_5_14B_INSTRUCT_1M_BF16, config=config)
    
    prompt = "Explain the concept of machine learning in simple terms."
    response = model.generate(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print()


def batch_processing():
    """Batch processing multiple prompts"""
    print("=== Batch Processing ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(LLMs.MISTRAL_NEMO_INSTRUCT_2407_BF16)
    
    prompts = [
        "What is Python?",
        "How does machine learning work?",
        "What are the benefits of renewable energy?",
        "Explain quantum computing briefly."
    ]
    
    print("Processing multiple prompts:")
    responses = model.batch_generate(prompts)
    
    for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
        print(f"{i}. Prompt: {prompt}")
        print(f"   Response: {response[:100]}...")
        print()


def model_comparison():
    """Compare different models on the same prompt"""
    print("=== Model Comparison ===")
    
    driver = UnifiedModelDriver()
    
    models = [
        LLMs.PHI_3_5_MINI_INSTRUCT_4BIT,
        LLMs.SMOLLM_1_7B_FP16,
        LLMs.MISTRAL_NEMO_INSTRUCT_2407_BF16
    ]
    
    prompt = "What is the meaning of life?"
    
    print(f"Comparing models on prompt: {prompt}")
    print()
    
    results = driver.compare_models(models, prompt)
    
    for model_name, response in results.items():
        print(f"Model: {model_name}")
        print(f"Response: {response[:150]}...")
        print("-" * 50)


def conversation_example():
    """Example of maintaining conversation context"""
    print("=== Conversation Example ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(LLMs.QWEN2_5_14B_INSTRUCT_1M_BF16)
    
    # System prompt for conversation
    system_prompt = "You are a helpful assistant. Keep responses concise but informative."
    
    conversation = [
        "Hello! What's your name?",
        "Can you help me with Python programming?",
        "What's the best way to learn data structures?",
        "Thanks for the help!"
    ]
    
    print("Conversation:")
    for i, user_input in enumerate(conversation, 1):
        response = model.generate(user_input, system_prompt=system_prompt)
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
        print()


def main():
    """Run all examples"""
    print("MLX Multimodal Toolkit - Text Generation Examples")
    print("=" * 50)
    print()
    
    try:
        basic_text_generation()
        streaming_generation()
        custom_configuration()
        batch_processing()
        model_comparison()
        conversation_example()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())