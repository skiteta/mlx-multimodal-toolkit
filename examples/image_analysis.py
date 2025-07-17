#!/usr/bin/env python3
"""
Examples for image analysis using Vision Language Models (VLMs)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_driver import UnifiedModelDriver
from src.models import VLMs


def basic_image_analysis():
    """Basic image analysis example"""
    print("=== Basic Image Analysis ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(VLMs.SMOLVLM_INSTRUCT_BF16)
    
    # Example image analysis
    image_path = "example.jpg"  # Replace with actual image path
    prompt = "What do you see in this image? Describe it in detail."
    
    try:
        response = model.generate(prompt, images=[image_path])
        print(f"Image: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print()
    except Exception as e:
        print(f"Error analyzing image: {e}")
        print("Make sure the image file exists and is accessible.")
        print()


def multiple_image_analysis():
    """Analyze multiple images"""
    print("=== Multiple Image Analysis ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(VLMs.FLORENCE_2_LARGE_FT_BF16)
    
    # Multiple images
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    prompt = "Compare these images and describe the differences."
    
    try:
        response = model.generate(prompt, images=image_paths)
        print(f"Images: {', '.join(image_paths)}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print()
    except Exception as e:
        print(f"Error analyzing images: {e}")
        print("Make sure all image files exist and are accessible.")
        print()


def image_question_answering():
    """Image-based question answering"""
    print("=== Image Question Answering ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(VLMs.QWEN2_5_VL_32B_INSTRUCT_BF16)
    
    image_path = "scene.jpg"
    questions = [
        "How many people are in this image?",
        "What is the weather like?",
        "What objects can you identify?",
        "What is the main activity happening?"
    ]
    
    print(f"Analyzing image: {image_path}")
    print()
    
    for i, question in enumerate(questions, 1):
        try:
            response = model.generate(question, images=[image_path])
            print(f"Q{i}: {question}")
            print(f"A{i}: {response}")
            print()
        except Exception as e:
            print(f"Error with question {i}: {e}")
            print()


def ocr_example():
    """Text extraction from images (OCR)"""
    print("=== OCR Example ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(VLMs.OLMOCR_7B_0225_PREVIWE_BF16)
    
    image_path = "document.jpg"
    prompt = "Extract all text from this image and format it properly."
    
    try:
        response = model.generate(prompt, images=[image_path])
        print(f"Image: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"Extracted text: {response}")
        print()
    except Exception as e:
        print(f"Error extracting text: {e}")
        print("Make sure the image contains readable text.")
        print()


def batch_image_processing():
    """Process multiple images in batch"""
    print("=== Batch Image Processing ===")
    
    driver = UnifiedModelDriver()
    
    # Prepare batch inputs
    inputs = [
        {"prompt": "What is the main subject?", "images": ["photo1.jpg"]},
        {"prompt": "Describe the colors in this image", "images": ["photo2.jpg"]},
        {"prompt": "What time of day is this?", "images": ["photo3.jpg"]},
        {"prompt": "Count the objects in this image", "images": ["photo4.jpg"]}
    ]
    
    try:
        responses = driver.batch_process(inputs, VLMs.MOLMO_7B_D_0924_BF16)
        
        for i, (input_data, response) in enumerate(zip(inputs, responses), 1):
            print(f"Batch {i}:")
            print(f"  Image: {input_data['images'][0]}")
            print(f"  Prompt: {input_data['prompt']}")
            print(f"  Response: {response}")
            print()
            
    except Exception as e:
        print(f"Error in batch processing: {e}")
        print()


def model_comparison():
    """Compare different VLM models"""
    print("=== VLM Model Comparison ===")
    
    driver = UnifiedModelDriver()
    
    models = [
        VLMs.FLORENCE_2_LARGE_FT_BF16,
        VLMs.SMOLVLM_INSTRUCT_BF16,
        VLMs.MOLMO_7B_D_0924_BF16
    ]
    
    image_path = "test_image.jpg"
    prompt = "Describe this image in detail."
    
    print(f"Comparing VLM models on image: {image_path}")
    print(f"Prompt: {prompt}")
    print()
    
    try:
        results = driver.compare_models(models, prompt, images=[image_path])
        
        for model_name, response in results.items():
            print(f"Model: {model_name}")
            print(f"Response: {response[:200]}...")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error comparing models: {e}")
        print()


def streaming_image_analysis():
    """Stream image analysis response"""
    print("=== Streaming Image Analysis ===")
    
    driver = UnifiedModelDriver()
    model = driver.load_model(VLMs.QWEN2_5_VL_32B_INSTRUCT_BF16)
    
    image_path = "complex_scene.jpg"
    prompt = "Provide a detailed analysis of this image, including objects, people, setting, and mood."
    
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print("Response (streaming): ", end="")
    
    try:
        for token in model.stream(prompt, images=[image_path]):
            print(token, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"Error streaming analysis: {e}")
        print()


def main():
    """Run all image analysis examples"""
    print("MLX Multimodal Toolkit - Image Analysis Examples")
    print("=" * 50)
    print()
    
    print("Note: Make sure you have sample images in the current directory")
    print("or update the image paths in the examples.")
    print()
    
    try:
        basic_image_analysis()
        multiple_image_analysis()
        image_question_answering()
        ocr_example()
        batch_image_processing()
        model_comparison()
        streaming_image_analysis()
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())