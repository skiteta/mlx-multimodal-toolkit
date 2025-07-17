#!/usr/bin/env python3
"""
Examples for combined multimodal processing pipelines
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.unified_driver import UnifiedModelDriver
from src.models import LLMs, VLMs, ALMs


def image_to_text_pipeline():
    """Image analysis followed by text processing"""
    print("=== Image to Text Pipeline ===")
    
    driver = UnifiedModelDriver()
    
    # Load VLM for image analysis
    vlm = driver.load_model(VLMs.QWEN2_5_VL_32B_INSTRUCT_BF16)
    
    # Load LLM for text processing
    llm = driver.load_model(LLMs.QWEN2_5_32B_INSTRUCT_BF16)
    
    image_path = "scene.jpg"
    
    try:
        # Step 1: Analyze image
        print("Step 1: Analyzing image...")
        image_description = vlm.generate(
            "Describe this image in detail, including objects, people, setting, and actions.",
            images=[image_path]
        )
        print(f"Image description: {image_description}")
        print()
        
        # Step 2: Generate story from description
        print("Step 2: Generating story from description...")
        story_prompt = f"Write a short creative story based on this scene: {image_description}"
        story = llm.generate(story_prompt, max_tokens=300)
        print(f"Generated story: {story}")
        print()
        
        # Step 3: Summarize the story
        print("Step 3: Summarizing the story...")
        summary_prompt = f"Summarize this story in one sentence: {story}"
        summary = llm.generate(summary_prompt, max_tokens=50)
        print(f"Story summary: {summary}")
        print()
        
    except Exception as e:
        print(f"Error in image-to-text pipeline: {e}")
        print()


def audio_to_summary_pipeline():
    """Audio transcription followed by summarization"""
    print("=== Audio to Summary Pipeline ===")
    
    driver = UnifiedModelDriver()
    
    # Load ALM for transcription
    alm = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)
    
    # Load LLM for summarization
    llm = driver.load_model(LLMs.QWEN2_5_14B_INSTRUCT_1M_BF16)
    
    audio_file = "meeting_recording.wav"
    
    try:
        # Step 1: Transcribe audio
        print("Step 1: Transcribing audio...")
        transcription = alm.generate(audio_file=audio_file)
        print(f"Transcription: {transcription[:200]}...")
        print()
        
        # Step 2: Extract key points
        print("Step 2: Extracting key points...")
        key_points_prompt = f"Extract the main points from this meeting transcript: {transcription}"
        key_points = llm.generate(key_points_prompt, max_tokens=300)
        print(f"Key points: {key_points}")
        print()
        
        # Step 3: Generate action items
        print("Step 3: Generating action items...")
        action_items_prompt = f"Based on this transcript, list specific action items: {transcription}"
        action_items = llm.generate(action_items_prompt, max_tokens=200)
        print(f"Action items: {action_items}")
        print()
        
    except Exception as e:
        print(f"Error in audio-to-summary pipeline: {e}")
        print()


def multimedia_content_analysis():
    """Analyze multiple types of content together"""
    print("=== Multimedia Content Analysis ===")
    
    driver = UnifiedModelDriver()
    
    # Load different models
    vlm = driver.load_model(VLMs.FLORENCE_2_LARGE_FT_BF16)
    alm = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)
    llm = driver.load_model(LLMs.MISTRAL_NEMO_INSTRUCT_2407_BF16)
    
    image_path = "presentation_slide.jpg"
    audio_file = "presentation_audio.wav"
    
    try:
        # Analyze image content
        print("Analyzing image content...")
        image_analysis = vlm.generate(
            "What information is presented in this slide? List key points.",
            images=[image_path]
        )
        print(f"Image analysis: {image_analysis}")
        print()
        
        # Transcribe audio content
        print("Transcribing audio content...")
        audio_transcription = alm.generate(audio_file=audio_file)
        print(f"Audio transcription: {audio_transcription[:200]}...")
        print()
        
        # Combine and analyze
        print("Combining multimodal analysis...")
        combined_prompt = f"""
        Analyze this presentation based on both visual and audio content:
        
        Visual content: {image_analysis}
        Audio content: {audio_transcription}
        
        Provide a comprehensive analysis of the presentation's effectiveness and key messages.
        """
        
        combined_analysis = llm.generate(combined_prompt, max_tokens=400)
        print(f"Combined analysis: {combined_analysis}")
        print()
        
    except Exception as e:
        print(f"Error in multimedia analysis: {e}")
        print()


def interactive_multimodal_chat():
    """Interactive chat with multimodal capabilities"""
    print("=== Interactive Multimodal Chat ===")
    
    driver = UnifiedModelDriver()
    
    # Load models
    vlm = driver.load_model(VLMs.SMOLVLM_INSTRUCT_BF16)
    alm = driver.load_model(ALMs.WHISPER_LARGE_V3_MLX)
    llm = driver.load_model(LLMs.PHI_3_5_MINI_INSTRUCT_4BIT)
    
    print("Multimodal Chat System (simulation)")
    print("Commands: 'image <path>', 'audio <path>', 'text <message>', 'quit'")
    print()
    
    # Simulate conversation
    conversation_history = []
    
    simulate_inputs = [
        ("image", "photo.jpg"),
        ("text", "What do you see in this image?"),
        ("audio", "question.wav"),
        ("text", "Can you elaborate on that?")
    ]
    
    for input_type, content in simulate_inputs:
        try:
            if input_type == "image":
                print(f"User: [Uploaded image: {content}]")
                response = vlm.generate(
                    "Describe this image briefly.",
                    images=[content]
                )
                conversation_history.append(f"Image: {response}")
                
            elif input_type == "audio":
                print(f"User: [Audio message: {content}]")
                transcription = alm.generate(audio_file=content)
                response = llm.generate(transcription)
                conversation_history.append(f"Audio transcription: {transcription}")
                conversation_history.append(f"Response: {response}")
                
            elif input_type == "text":
                print(f"User: {content}")
                # Include conversation history for context
                context = "\n".join(conversation_history[-3:])  # Last 3 exchanges
                full_prompt = f"Context: {context}\n\nUser: {content}\n\nAssistant:"
                response = llm.generate(full_prompt, max_tokens=150)
                conversation_history.append(f"User: {content}")
                conversation_history.append(f"Assistant: {response}")
                
            print(f"Assistant: {response}")
            print()
            
        except Exception as e:
            print(f"Error processing {input_type}: {e}")
            print()


def content_generation_pipeline():
    """Generate content across multiple modalities"""
    print("=== Content Generation Pipeline ===")
    
    driver = UnifiedModelDriver()
    
    llm = driver.load_model(LLMs.QWEN2_5_CODER_32B_INSTRUCT_BF16)
    vlm = driver.load_model(VLMs.MOLMO_7B_D_0924_BF16)
    
    try:
        # Step 1: Generate text content
        print("Step 1: Generating blog post content...")
        blog_prompt = "Write a blog post about the future of AI in education."
        blog_content = llm.generate(blog_prompt, max_tokens=500)
        print(f"Blog content: {blog_content[:300]}...")
        print()
        
        # Step 2: Generate image description for illustrations
        print("Step 2: Generating image suggestions...")
        image_prompt = f"Based on this blog post, suggest 3 images that would illustrate the content: {blog_content}"
        image_suggestions = llm.generate(image_prompt, max_tokens=200)
        print(f"Image suggestions: {image_suggestions}")
        print()
        
        # Step 3: Create social media summary
        print("Step 3: Creating social media summary...")
        social_prompt = f"Create a Twitter-length summary of this blog post: {blog_content}"
        social_summary = llm.generate(social_prompt, max_tokens=50)
        print(f"Social media summary: {social_summary}")
        print()
        
    except Exception as e:
        print(f"Error in content generation pipeline: {e}")
        print()


def main():
    """Run all multimodal pipeline examples"""
    print("MLX Multimodal Toolkit - Multimodal Pipeline Examples")
    print("=" * 60)
    print()
    
    print("Note: These examples demonstrate how to combine different")
    print("model types for complex multimodal tasks.")
    print()
    
    try:
        image_to_text_pipeline()
        audio_to_summary_pipeline()
        multimedia_content_analysis()
        interactive_multimodal_chat()
        content_generation_pipeline()
        
    except Exception as e:
        print(f"Error running pipeline examples: {e}")
        print("Make sure you have the required dependencies installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())