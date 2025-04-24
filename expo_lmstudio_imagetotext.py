"""
@author: Matt John Powell
@title: LM Studio Nodes for ComfyUI
@nickname: LM Studio Nodes
@description: This extension provides three custom nodes for ComfyUI that integrate LM Studio's capabilities:
1. Image to Text: Generates text descriptions of images using vision models.
2. Text Generation: Generates text based on a given prompt using language models.
3. Unified Node: A versatile node that can handle both image and text inputs.
All nodes leverage the LM Studio Python SDK for better performance and reliability.
"""

import base64
import numpy as np
from PIL import Image
import io
import random
import time
import os
import tempfile

# Default models to use
DEFAULT_LLM = "gemma-3-4b-it-qat"
DEFAULT_VISION = "qwen2-vl-2b-instruct"

# Try to import LM Studio SDK
try:
    import lmstudio as lms
    HAS_SDK = True
    print("LM Studio SDK found and loaded")
except ImportError:
    lms = None
    HAS_SDK = False
    print("LM Studio SDK not found. Please install it using: pip install lmstudio")

class ExpoLmstudioUnified:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"default": "give me a prompt for an image generation"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "model_key": ("STRING", {"default": DEFAULT_LLM}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "process_input"
    CATEGORY = "ComfyExpo/LMStudio"
    
    def IS_CHANGED(self, **kwargs):
        return float("NaN")  # Tell ComfyUI to process this node the usual way

    def process_input(self, text_input, system_prompt, model_key, auto_unload, unload_delay, seed, image=None, max_tokens=1000, temperature=0.7, debug=False):
        # Check if LM Studio SDK is available
        if not HAS_SDK:
            return ("Error: LM Studio SDK is not installed. Please install it using: pip install lmstudio",)
        
        # Check if we have valid inputs
        has_image = image is not None
        has_text = text_input.strip() != ""
        
        # If no inputs are provided, return a message
        if not has_image and not has_text:
            return ("No inputs provided. Please connect an image or provide text input.",)
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)
        
        if debug:
            print(f"Debug: Starting unified process_input method")
            print(f"Debug: Has image input: {has_image}")
            print(f"Debug: Has text input: {has_text}")
            print(f"Debug: Model: {model_key}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")

        try:
            # Get model reference with TTL if auto_unload is enabled
            start_time = time.time()
            
            if auto_unload == "True" and unload_delay > 0:
                # Use TTL for delayed unloading
                model = lms.llm(model_key, ttl=unload_delay)
                if debug:
                    print(f"Debug: Model loaded with TTL={unload_delay}s in {time.time() - start_time:.2f}s")
            else:
                # Normal loading
                model = lms.llm(model_key)
                if debug:
                    print(f"Debug: Model loaded in {time.time() - start_time:.2f}s")
            
            # Create a new chat
            chat = lms.Chat(system_prompt)
            
            # Process inputs
            if has_image and has_text:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(np.uint8(image[0]*255))
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    # Save to the temporary file
                    pil_image.save(temp_path, format="JPEG")
                
                try:
                    if debug:
                        print(f"Debug: Saved image to temporary file: {temp_path}")
                    
                    # Use the file path method to prepare the image
                    image_handle = lms.prepare_image(temp_path)
                    
                    # Add user message with both text and image
                    chat.add_user_message(text_input, images=[image_handle])
                    
                    if debug:
                        print(f"Debug: Added image to chat message")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        if debug:
                            print(f"Debug: Removed temporary file: {temp_path}")
            
            elif has_image:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(np.uint8(image[0]*255))
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    # Save to the temporary file
                    pil_image.save(temp_path, format="JPEG")
                
                try:
                    if debug:
                        print(f"Debug: Saved image to temporary file: {temp_path}")
                    
                    # Use the file path method to prepare the image
                    image_handle = lms.prepare_image(temp_path)
                    
                    # Add user message with image only
                    chat.add_user_message("Analyze this image:", images=[image_handle])
                    
                    if debug:
                        print(f"Debug: Added image to chat message")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        if debug:
                            print(f"Debug: Removed temporary file: {temp_path}")
            
            elif has_text:
                # Add user message with text only
                chat.add_user_message(text_input)
            
            # Configure generation parameters
            config = {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "seed": seed
            }
            
            if debug:
                print(f"Debug: Sending request to LM Studio with config: {config}")
            
            # Generate response
            result = model.respond(chat, config=config)
            
            if debug:
                print(f"Debug: Response received: {result.content[:100]}...")  # Print first 100 characters
                print(f"Debug: Tokens generated: {result.stats.predicted_tokens_count}")
                print(f"Debug: Generation time: {result.stats.generation_time_sec}s")
            
            # Unload model immediately if requested
            if auto_unload == "True" and unload_delay == 0:
                try:
                    if debug:
                        print(f"Debug: Unloading model immediately: {model_key}")
                    model.unload()
                except Exception as unload_err:
                    print(f"Warning: Failed to unload model: {unload_err}")
            
            return (result.content,)

        except Exception as e:
            error_message = f"Error processing with LM Studio: {str(e)}"
            print(error_message)
            return (error_message,)

class ExpoLmstudioImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "Describe this image in detail"}),
                "system_prompt": ("STRING", {"default": "This is a chat between a user and an assistant. The assistant is an expert in describing images, with detail and accuracy"}),
                "model_key": ("STRING", {"default": DEFAULT_VISION}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)
    FUNCTION = "process_image"
    CATEGORY = "ComfyExpo/I2T"
    
    def IS_CHANGED(self, **kwargs):
        return float("NaN")  # Tell ComfyUI to process this node the usual way

    def process_image(self, image, user_prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False):
        # Check if LM Studio SDK is available
        if not HAS_SDK:
            return ("Error: LM Studio SDK is not installed. Please install it using: pip install lmstudio",)
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)
        
        if debug:
            print(f"Debug: Starting process_image method")
            print(f"Debug: Text input: {user_prompt}")
            print(f"Debug: Model: {model_key}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")
            print(f"Debug: Image shape: {image.shape}")

        try:
            # Get model reference with TTL if auto_unload is enabled
            start_time = time.time()
            
            if auto_unload == "True" and unload_delay > 0:
                # Use TTL for delayed unloading
                model_obj = lms.llm(model_key, ttl=unload_delay)
                if debug:
                    print(f"Debug: Model loaded with TTL={unload_delay}s in {time.time() - start_time:.2f}s")
            else:
                # Normal loading
                model_obj = lms.llm(model_key)
                if debug:
                    print(f"Debug: Model loaded in {time.time() - start_time:.2f}s")
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(np.uint8(image[0]*255))
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                # Save to the temporary file
                pil_image.save(temp_path, format="JPEG")
            
            try:
                if debug:
                    print(f"Debug: Saved image to temporary file: {temp_path}")
                
                # Use the file path method to prepare the image
                image_handle = lms.prepare_image(temp_path)
                
                # Create a new chat
                chat = lms.Chat(system_prompt)
                
                # Add user message with image
                chat.add_user_message(user_prompt, images=[image_handle])
                
                if debug:
                    print(f"Debug: Added image to chat message")
                
                # Configure generation parameters
                config = {
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "seed": seed
                }
                
                if debug:
                    print(f"Debug: Sending request to LM Studio")
                
                # Generate response
                result = model_obj.respond(chat, config=config)
                
                if debug:
                    print(f"Debug: Response received: {result.content[:100]}...")  # Print first 100 characters
                    print(f"Debug: Tokens generated: {result.stats.predicted_tokens_count}")
                    print(f"Debug: Generation time: {result.stats.generation_time_sec}s")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    if debug:
                        print(f"Debug: Removed temporary file: {temp_path}")
            
            # Unload model immediately if requested
            if auto_unload == "True" and unload_delay == 0:
                try:
                    if debug:
                        print(f"Debug: Unloading model immediately: {model_key}")
                    model_obj.unload()
                except Exception as unload_err:
                    print(f"Warning: Failed to unload model: {unload_err}")
            
            return (result.content,)

        except Exception as e:
            error_message = f"Error processing with LM Studio: {str(e)}"
            print(error_message)
            return (error_message,)
        
class ExpoLmstudioTextGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate a creative story:"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "model_key": ("STRING", {"default": DEFAULT_LLM}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "unload_delay": ("INT", {"default": 0, "min": 0, "max": 3600, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "generate_text"
    CATEGORY = "ComfyExpo/Text"
    
    def IS_CHANGED(self, **kwargs):
        return float("NaN")  # Tell ComfyUI to process this node the usual way

    def generate_text(self, prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False):
        # Check if LM Studio SDK is available
        if not HAS_SDK:
            return ("Error: LM Studio SDK is not installed. Please install it using: pip install lmstudio",)
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting generate_text method")
            print(f"Debug: Prompt: {prompt}")
            print(f"Debug: Model: {model_key}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")
            print(f"Debug: Max tokens: {max_tokens}")
            print(f"Debug: Temperature: {temperature}")

        try:
            # Get model reference with TTL if auto_unload is enabled
            start_time = time.time()
            
            if auto_unload == "True" and unload_delay > 0:
                # Use TTL for delayed unloading
                model_obj = lms.llm(model_key, ttl=unload_delay)
                if debug:
                    print(f"Debug: Model loaded with TTL={unload_delay}s in {time.time() - start_time:.2f}s")
            else:
                # Normal loading
                model_obj = lms.llm(model_key)
                if debug:
                    print(f"Debug: Model loaded in {time.time() - start_time:.2f}s")
            
            # Create a new chat
            chat = lms.Chat(system_prompt)
            
            # Add user message
            chat.add_user_message(prompt)
            
            # Configure generation parameters
            config = {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "seed": seed
            }
            
            if debug:
                print(f"Debug: Sending request to LM Studio")
            
            # Generate response
            result = model_obj.respond(chat, config=config)
            
            if debug:
                print(f"Debug: Response received: {result.content[:100]}...")  # Print first 100 characters
                print(f"Debug: Tokens generated: {result.stats.predicted_tokens_count}")
                print(f"Debug: Generation time: {result.stats.generation_time_sec}s")
            
            # Unload model immediately if requested
            if auto_unload == "True" and unload_delay == 0:
                try:
                    if debug:
                        print(f"Debug: Unloading model immediately: {model_key}")
                    model_obj.unload()
                except Exception as unload_err:
                    print(f"Warning: Failed to unload model: {unload_err}")
            
            return (result.content,)

        except Exception as e:
            error_message = f"Error processing with LM Studio: {str(e)}"
            print(error_message)
            return (error_message,)