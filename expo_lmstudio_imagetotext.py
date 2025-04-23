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
import importlib.util
import sys
import os
import time

# Cache for available models - populated when modules are accessed
AVAILABLE_MODELS = {
    "llm": [],
    "vision": []
}

# Default models to use when nothing is available
DEFAULT_LLM = "gemma-3-4b-it-qat"
DEFAULT_VISION = "qwen2-vl-2b-instruct"

# Flag to track if we've attempted to load models
MODELS_LOADED = False

# Function to load/refresh available models
def refresh_available_models():
    global AVAILABLE_MODELS, MODELS_LOADED
    
    # Mark that we've attempted to load models
    MODELS_LOADED = True
    
    # Clear existing models
    AVAILABLE_MODELS = {
        "llm": [],
        "vision": []
    }
    
    # Try to import LM Studio SDK
    try:
        import lmstudio as lms
        
        try:
            # Get all models
            all_models = lms.list_downloaded_models("llm")
            
            # Separate regular LLMs from vision models
            for model in all_models:
                AVAILABLE_MODELS["llm"].append({
                    "key": model.model_key,
                    "name": model.display_name
                })
                
                # If model has vision capability, add to vision models too
                if hasattr(model, 'vision') and model.vision:
                    AVAILABLE_MODELS["vision"].append({
                        "key": model.model_key,
                        "name": model.display_name
                    })
            
            print(f"LM Studio SDK loaded - Found {len(AVAILABLE_MODELS['llm'])} LLM models and {len(AVAILABLE_MODELS['vision'])} vision models")
            return True
            
        except Exception as e:
            print(f"Error loading LM Studio models: {str(e)}")
            return False
            
    except ImportError:
        print("LM Studio SDK not found. Please install it using: pip install lmstudio")
        return False

# Try to import LM Studio SDK
try:
    import lmstudio as lms
    HAS_SDK = True
except ImportError:
    lms = None
    HAS_SDK = False
    print("LM Studio SDK not found. Please install it using: pip install lmstudio")

class ExpoLmstudioUnified:
    @classmethod
    def INPUT_TYPES(cls):
        global MODELS_LOADED
        
        # Load models if not loaded yet
        if not MODELS_LOADED:
            refresh_available_models()
        
        # Get model options from our cache
        llm_model_keys = [model["key"] for model in AVAILABLE_MODELS["llm"]]
        
        # Add default options if no models found
        if not llm_model_keys:
            llm_model_keys = [DEFAULT_LLM]
        
        return {
            "required": {
                "model_selection": (["DROPDOWN", "CUSTOM"], {"default": "DROPDOWN"}),
                "model_key": (llm_model_keys, {"default": llm_model_keys[0] if llm_model_keys else DEFAULT_LLM}),
                "custom_model": ("STRING", {"default": DEFAULT_LLM}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "text_input": ("STRING", {"default": "give me a prompt for an image generation"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "process_input"
    CATEGORY = "ComfyExpo/LMStudio"

    def process_input(self, model_selection, model_key, custom_model, system_prompt, auto_unload, seed, image=None, text_input="", max_tokens=1000, temperature=0.7, debug=False, enabled=True):
        # Skip processing if disabled
        if not enabled:
            return ("Node is disabled. Enable it to process inputs.",)
        
        # Check if LM Studio SDK is available
        if not HAS_SDK:
            return ("Error: LM Studio SDK is not installed. Please install it using: pip install lmstudio",)
        
        # Check if we have valid inputs
        has_image = image is not None
        has_text = text_input.strip() != ""
        
        # If no inputs are provided, return a message
        if not has_image and not has_text:
            return ("No inputs provided. Please connect an image or provide text input.",)
        
        # Determine which model to use
        selected_model = custom_model if model_selection == "CUSTOM" else model_key
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)
        
        if debug:
            print(f"Debug: Starting unified process_input method")
            print(f"Debug: Has image input: {has_image}")
            print(f"Debug: Has text input: {has_text}")
            print(f"Debug: Model source: {model_selection}")
            print(f"Debug: Selected model: {selected_model}")
            print(f"Debug: System prompt: {system_prompt}")

        try:
            # Get model reference
            start_time = time.time()
            model = lms.llm(selected_model)
            if debug:
                print(f"Debug: Model loaded in {time.time() - start_time:.2f}s")
            
            # Create a new chat
            chat = lms.Chat(system_prompt)
            
            # Process inputs
            if has_image and has_text:
                # Process image to prepare for SDK
                pil_image = Image.fromarray(np.uint8(image[0]*255))
                
                # Save image to temporary memory buffer
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                image_bytes = buffered.getvalue()
                
                # Prepare image with SDK
                image_handle = lms.prepare_image(image_bytes)
                
                # Add user message with both text and image
                chat.add_user_message(text_input, images=[image_handle])
            
            elif has_image:
                # Process image only
                pil_image = Image.fromarray(np.uint8(image[0]*255))
                
                # Save image to temporary memory buffer
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                image_bytes = buffered.getvalue()
                
                # Prepare image with SDK
                image_handle = lms.prepare_image(image_bytes)
                
                # Add user message with image only
                chat.add_user_message("Analyze this image:", images=[image_handle])
            
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
            
            # Unload model if requested
            if auto_unload == "True":
                try:
                    if debug:
                        print(f"Debug: Unloading model: {selected_model}")
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
        global MODELS_LOADED
        
        # Load models if not loaded yet
        if not MODELS_LOADED:
            refresh_available_models()
        
        # Get model options from our cache
        vision_model_keys = [model["key"] for model in AVAILABLE_MODELS["vision"]]
        
        # Add default options if no models found
        if not vision_model_keys:
            vision_model_keys = [DEFAULT_VISION]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "Describe this image in detail"}),
                "model_selection": (["DROPDOWN", "CUSTOM"], {"default": "DROPDOWN"}),
                "model": (vision_model_keys, {"default": vision_model_keys[0] if vision_model_keys else DEFAULT_VISION}),
                "custom_model": ("STRING", {"default": DEFAULT_VISION}),
                "system_prompt": ("STRING", {"default": "This is a chat between a user and an assistant. The assistant is an expert in describing images, with detail and accuracy"}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)
    FUNCTION = "process_image"
    CATEGORY = "ComfyExpo/I2T"

    def process_image(self, image, user_prompt, model_selection, model, custom_model, system_prompt, auto_unload, seed, max_tokens=1000, temperature=0.7, debug=False, enabled=True):
        # Skip processing if disabled
        if not enabled:
            return ("Node is disabled. Enable it to process inputs.",)
        
        # Check if LM Studio SDK is available
        if not HAS_SDK:
            return ("Error: LM Studio SDK is not installed. Please install it using: pip install lmstudio",)
        
        # Determine which model to use
        selected_model = custom_model if model_selection == "CUSTOM" else model
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)
        
        if debug:
            print(f"Debug: Starting process_image method")
            print(f"Debug: Text input: {user_prompt}")
            print(f"Debug: Model source: {model_selection}")
            print(f"Debug: Selected model: {selected_model}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Image shape: {image.shape}")

        try:
            # Get model reference
            start_time = time.time()
            model_obj = lms.llm(selected_model)
            if debug:
                print(f"Debug: Model loaded in {time.time() - start_time:.2f}s")
            
            # Process image to prepare for SDK
            pil_image = Image.fromarray(np.uint8(image[0]*255))
            
            # Save image to temporary memory buffer
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            image_bytes = buffered.getvalue()
            
            # Prepare image with SDK
            image_handle = lms.prepare_image(image_bytes)
            
            # Create a new chat
            chat = lms.Chat(system_prompt)
            
            # Add user message with image
            chat.add_user_message(user_prompt, images=[image_handle])
            
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
            
            # Unload model if requested
            if auto_unload == "True":
                try:
                    if debug:
                        print(f"Debug: Unloading model: {selected_model}")
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
        global MODELS_LOADED
        
        # Load models if not loaded yet
        if not MODELS_LOADED:
            refresh_available_models()
        
        # Get model options from our cache
        llm_model_keys = [model["key"] for model in AVAILABLE_MODELS["llm"]]
        
        # Add default options if no models found
        if not llm_model_keys:
            llm_model_keys = [DEFAULT_LLM]
        
        return {
            "required": {
                "prompt": ("STRING", {"default": "Generate a creative story:"}),
                "model_selection": (["DROPDOWN", "CUSTOM"], {"default": "DROPDOWN"}),
                "model": (llm_model_keys, {"default": llm_model_keys[0] if llm_model_keys else DEFAULT_LLM}),
                "custom_model": ("STRING", {"default": DEFAULT_LLM}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "auto_unload": (["True", "False"], {"default": "True"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "generate_text"
    CATEGORY = "ComfyExpo/Text"

    def generate_text(self, prompt, model_selection, model, custom_model, system_prompt, auto_unload, seed, max_tokens=1000, temperature=0.7, debug=False, enabled=True):
        # Skip processing if disabled
        if not enabled:
            return ("Node is disabled. Enable it to process inputs.",)
        
        # Check if LM Studio SDK is available
        if not HAS_SDK:
            return ("Error: LM Studio SDK is not installed. Please install it using: pip install lmstudio",)
        
        # Determine which model to use
        selected_model = custom_model if model_selection == "CUSTOM" else model
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting generate_text method")
            print(f"Debug: Prompt: {prompt}")
            print(f"Debug: Model source: {model_selection}")
            print(f"Debug: Selected model: {selected_model}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Max tokens: {max_tokens}")
            print(f"Debug: Temperature: {temperature}")

        try:
            # Get model reference
            start_time = time.time()
            model_obj = lms.llm(selected_model)
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
            
            # Unload model if requested
            if auto_unload == "True":
                try:
                    if debug:
                        print(f"Debug: Unloading model: {selected_model}")
                    model_obj.unload()
                except Exception as unload_err:
                    print(f"Warning: Failed to unload model: {unload_err}")
            
            return (result.content,)

        except Exception as e:
            error_message = f"Error processing with LM Studio: {str(e)}"
            print(error_message)
            return (error_message,)

class ExpoLmstudioSetup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["INSTALL SDK", "REFRESH MODELS", "SDK STATUS"], {"default": "SDK STATUS"}),
                "trigger": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, "display": "number"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "setup_action"
    CATEGORY = "ComfyExpo/LMStudio"
    
    def setup_action(self, action, trigger):
        global AVAILABLE_MODELS, MODELS_LOADED
        result = ""
        
        if action == "INSTALL SDK":
            try:
                import subprocess
                process = subprocess.Popen(
                    ["pip", "install", "lmstudio"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    result = "Successfully installed LM Studio SDK.\n\n" + stdout
                    
                    # Try to import the newly installed SDK
                    try:
                        # Update the global SDK status
                        global HAS_SDK
                        
                        # Try to load the module
                        if 'lmstudio' in sys.modules:
                            import importlib
                            importlib.reload(sys.modules['lmstudio'])
                        
                        global lms
                        import lmstudio as lms
                        HAS_SDK = True
                        result += "\n\nLM Studio SDK loaded successfully."
                        
                        # Refresh the model list
                        refresh_available_models()
                    except Exception as import_err:
                        result += f"\n\nError loading the SDK: {str(import_err)}"
                else:
                    result = f"Error installing LM Studio SDK:\n{stderr}"
            except Exception as e:
                result = f"Error: {str(e)}"
        
        elif action == "REFRESH MODELS":
            try:
                if not HAS_SDK:
                    return ("LM Studio SDK is not installed. Please install it first.",)
                
                # Call the refresh function
                if refresh_available_models():
                    result = f"Models refreshed successfully.\n\nFound {len(AVAILABLE_MODELS['llm'])} LLM models and {len(AVAILABLE_MODELS['vision'])} vision models.\n\n"
                    
                    # List the models
                    if AVAILABLE_MODELS["llm"]:
                        result += "LLM Models:\n"
                        for model in AVAILABLE_MODELS["llm"]:
                            result += f"- {model['name']} (Key: {model['key']})\n"
                        
                        result += "\n"
                    
                    if AVAILABLE_MODELS["vision"]:
                        result += "Vision Models:\n"
                        for model in AVAILABLE_MODELS["vision"]:
                            result += f"- {model['name']} (Key: {model['key']})\n"
                else:
                    result = "Failed to refresh models. Please check LM Studio is running correctly."
                
            except Exception as e:
                result = f"Error refreshing models: {str(e)}"
        
        elif action == "SDK STATUS":
            if not HAS_SDK:
                result = "LM Studio SDK is not installed.\n\nUse the 'INSTALL SDK' action to install it."
            else:
                result = "LM Studio SDK is installed and ready to use.\n\n"
                
                if not MODELS_LOADED:
                    result += "No models have been loaded yet. Use 'REFRESH MODELS' to scan for available models.\n\n"
                else:
                    result += f"Found {len(AVAILABLE_MODELS['llm'])} LLM models and {len(AVAILABLE_MODELS['vision'])} vision models.\n\n"
                    
                    # Show sample of models if available
                    if AVAILABLE_MODELS["llm"]:
                        result += "Sample LLM Models:\n"
                        for i, model in enumerate(AVAILABLE_MODELS["llm"][:5]):  # Show first 5 models
                            result += f"- {model['name']} (Key: {model['key']})\n"
                        
                        if len(AVAILABLE_MODELS["llm"]) > 5:
                            result += f"And {len(AVAILABLE_MODELS['llm']) - 5} more...\n"
                        
                        result += "\n"
                    
                    if AVAILABLE_MODELS["vision"]:
                        result += "Sample Vision Models:\n"
                        for i, model in enumerate(AVAILABLE_MODELS["vision"][:5]):  # Show first 5 models
                            result += f"- {model['name']} (Key: {model['key']})\n"
                        
                        if len(AVAILABLE_MODELS["vision"]) > 5:
                            result += f"And {len(AVAILABLE_MODELS['vision']) - 5} more...\n"
        
        return (result,)