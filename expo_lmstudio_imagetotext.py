"""
@author: Matt John Powell
@title: LM Studio Nodes for ComfyUI
@nickname: LM Studio Nodes
@description: This extension provides three custom nodes for ComfyUI that integrate LM Studio's capabilities:
1. Image to Text: Generates text descriptions of images using vision models.
2. Text Generation: Generates text based on a given prompt using language models.
3. Unified Node: A versatile node that can handle both image and text inputs.
All nodes leverage the LM Studio Python SDK for better performance and reliability.
Includes a fallback mechanism to use any loaded model if the specified model fails.
"""

import base64
import numpy as np
from PIL import Image
import io
import random
import time
import os
import tempfile
import concurrent.futures
import hashlib

# Default models to use
DEFAULT_LLM = "gemma-3-4b-it-qat"
DEFAULT_VISION = "qwen2-vl-2b-instruct"

# Try to import LM Studio SDK
try:
    import lmstudio as lms
    HAS_SDK = True
    print("LM Studio SDK found and loaded")
    # Note: Updated to use modern API (lms.list_loaded_models() instead of lms.server.get_loaded_models())
    # Compatible with LM Studio SDK version 1.4.1 and newer
except ImportError:
    lms = None
    HAS_SDK = False
    print("LM Studio SDK not found. Please install it using: pip install lmstudio")

def check_sdk_compatibility():
    """Check if the LM Studio SDK version is compatible with the current LM Studio version"""
    if not HAS_SDK:
        return False, "LM Studio SDK is not installed. Please install it using: pip install lmstudio"
    
    try:
        # Try a simple operation that would fail with version mismatch
        # Try both modern and legacy API approaches
        try:
            lms.list_loaded_models()
        except AttributeError:
            # Fallback to client approach if direct method doesn't exist
            with lms.Client() as client:
                client.list_loaded_models()
        return True, None
    except Exception as e:
        error_str = str(e)
        if "bosToken" in error_str or "jinjaPromptTemplate" in error_str:
            return False, ("LM Studio SDK version incompatibility detected. "
                          "Please upgrade the SDK using: pip install lmstudio --upgrade")
        return False, f"SDK compatibility check failed: {error_str}"

# --- Helper function to get model with fallback ---
def get_lm_model_with_fallback(model_key, auto_unload, unload_delay, debug=False):
    """
    Attempts to get the specified model. If it fails, tries to find and use
    any currently loaded model in LM Studio as a fallback.
    Returns the model object or raises an exception if no model can be obtained.
    """
    # First check SDK compatibility
    is_compatible, compatibility_error = check_sdk_compatibility()
    if not is_compatible:
        raise Exception(compatibility_error)
    
    model_obj = None
    try:
        start_time = time.time()
        if auto_unload == "True" and unload_delay > 0:
            # Use TTL for delayed unloading
            model_obj = lms.llm(model_key, ttl=unload_delay)
            if debug:
                print(f"Debug: Primary model '{model_key}' loaded with TTL={unload_delay}s in {time.time() - start_time:.2f}s")
        else:
            # Normal loading
            model_obj = lms.llm(model_key)
            if debug:
                print(f"Debug: Primary model '{model_key}' loaded in {time.time() - start_time:.2f}s")
        return model_obj

    except Exception as e:
        print(f"Warning: Failed to load or get primary model '{model_key}': {e}")
        print("Attempting to find a loaded model as fallback...")

        try:
            # Try to find any loaded models using modern API
            loaded_models = None
            try:
                loaded_models = lms.list_loaded_models()
            except AttributeError:
                # Fallback to client approach if direct method doesn't exist
                try:
                    with lms.Client() as client:
                        loaded_models = client.list_loaded_models()
                except:
                    pass

            if loaded_models:
                # Handle different data structures from different SDK versions
                fallback_model_key = None
                
                if isinstance(loaded_models, list) and len(loaded_models) > 0:
                    first_model = loaded_models[0]
                    if isinstance(first_model, dict):
                        # Legacy format: [{"model": "model_name", ...}, ...]
                        fallback_model_key = first_model.get('model') or first_model.get('id') or first_model.get('name')
                    elif isinstance(first_model, str):
                        # Simple format: ["model_name", ...]
                        fallback_model_key = first_model
                
                if not fallback_model_key:
                    raise Exception("Could not extract model name from loaded models list")
                
                print(f"Debug: Found loaded model '{fallback_model_key}'. Attempting to use as fallback.")

                # Load/get the fallback model, applying the same unload settings
                start_time = time.time()
                if auto_unload == "True" and unload_delay > 0:
                     model_obj = lms.llm(fallback_model_key, ttl=unload_delay)
                     if debug:
                         print(f"Debug: Fallback model '{fallback_model_key}' loaded with TTL={unload_delay}s in {time.time() - start_time:.2f}s")
                else:
                     model_obj = lms.llm(fallback_model_key)
                     if debug:
                         print(f"Debug: Fallback model '{fallback_model_key}' loaded in {time.time() - start_time:.2f}s")

                print(f"Info: Successfully obtained fallback model '{fallback_model_key}'.")
                return model_obj
            else:
                raise Exception(f"Failed to load primary model '{model_key}' and no other models are currently loaded in LM Studio.")

        except Exception as fallback_e:
            # If fallback also fails or no models are loaded
            raise Exception(f"Error: Failed to load primary model '{model_key}' and fallback attempt failed. Details: {fallback_e}")


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
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "process_input"
    CATEGORY = "ComfyExpo/LMStudio"

    @classmethod
    def IS_CHANGED(cls, text_input, system_prompt, model_key, auto_unload, unload_delay, seed, image=None, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300):
        m = hashlib.sha256()
        
        m.update(str(text_input).encode())
        m.update(str(system_prompt).encode())
        m.update(str(model_key).encode())
        m.update(str(auto_unload).encode())
        m.update(str(unload_delay).encode())
        m.update(str(seed).encode())
        m.update(str(max_tokens).encode())
        m.update(str(temperature).encode())
        m.update(str(debug).encode())
        m.update(str(timeout_seconds).encode())
        
        # Include image hash if present
        if image is not None:
            # Convert image to a hashable representation
            image_bytes = np.array(image).tobytes()
            m.update(image_bytes)
        
        return m.hexdigest()

    def process_input(self, text_input, system_prompt, model_key, auto_unload, unload_delay, seed, image=None, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300):
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
            print(f"Debug: Requested Model: {model_key}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")

        temp_path = None # Initialize temp_path for cleanup

        try:
            # --- Use helper function to get model with fallback ---
            model = get_lm_model_with_fallback(model_key, auto_unload, unload_delay, debug)
            chat = lms.Chat(system_prompt)

            # Process inputs
            if has_image:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(np.uint8(image[0]*255))

                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    # Save to the temporary file
                    pil_image.save(temp_path, format="JPEG")

                if debug:
                    print(f"Debug: Saved image to temporary file: {temp_path}")

                # Use the file path method to prepare the image
                image_handle = lms.prepare_image(temp_path)

                # Add user message with image and optional text
                user_message_content = [image_handle]
                if has_text:
                     user_message_content.insert(0, text_input) # Add text before image handle

                chat.add_user_message(*user_message_content)

                if debug:
                    print(f"Debug: Added image(s) and text (if any) to chat message")

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
            # --- Timeout logic ---
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.respond, chat, config=config)
                try:
                    result = future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    error_message = f"Error: LM Studio model response timed out after {timeout_seconds} seconds."
                    print(error_message)
                    return (error_message,)

            if debug:
                print(f"Debug: Response received: {result.content[:100]}...")  # Print first 100 characters
                print(f"Debug: Tokens generated: {result.stats.predicted_tokens_count}")
                print(f"Debug: Time to first token: {result.stats.time_to_first_token_sec}s")

            # Unload model immediately if requested (TTL handles delayed unloading)
            if auto_unload == "True" and unload_delay == 0:
                try:
                    if debug:
                        print(f"Debug: Unloading model immediately.")
                    model.unload()
                except Exception as unload_err:
                    print(f"Warning: Failed to unload model: {unload_err}")

            return (result.content,)

        except Exception as e:
            error_message = f"Error processing with LM Studio: {str(e)}"
            print(error_message)
            return (error_message,)
        finally:
            # Clean up the temporary image file if it was created
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    if debug:
                        print(f"Debug: Removed temporary file: {temp_path}")
                except Exception as cleanup_err:
                    print(f"Warning: Failed to remove temporary file {temp_path}: {cleanup_err}")


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
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 1}),
                # Legacy parameters for backward compatibility
                "model": ("STRING", {"default": ""}),  # Old parameter name
                "ip_address": ("STRING", {"default": ""}),  # Legacy HTTP mode
                "port": ("INT", {"default": 0, "min": 0, "max": 65535}),  # Legacy HTTP mode
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)
    FUNCTION = "process_image"
    CATEGORY = "ComfyExpo/I2T"

    @classmethod
    def IS_CHANGED(cls, image, user_prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, model="", ip_address="", port=0):
        m = hashlib.sha256()
        
        m.update(str(user_prompt).encode())
        m.update(str(system_prompt).encode())
        m.update(str(model_key).encode())
        m.update(str(auto_unload).encode())
        m.update(str(unload_delay).encode())
        m.update(str(seed).encode())
        m.update(str(max_tokens).encode())
        m.update(str(temperature).encode())
        m.update(str(debug).encode())
        m.update(str(timeout_seconds).encode())
        m.update(str(model).encode())
        m.update(str(ip_address).encode())
        m.update(str(port).encode())
        
        # Include image hash
        if image is not None:
            image_bytes = np.array(image).tobytes()
            m.update(image_bytes)
        
        return m.hexdigest()

    def process_image(self, image, user_prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, model="", ip_address="", port=0):
        # Handle backward compatibility
        # If legacy parameters are provided, show a deprecation warning and try to use them
        if model and not model_key:
            model_key = model
            if debug:
                print("Debug: Using legacy 'model' parameter as 'model_key'")
        
        if ip_address and port > 0:
            # Legacy HTTP mode detected
            return self._process_image_legacy_http(image, user_prompt, system_prompt, model_key or model, ip_address, port, seed, max_tokens, temperature, debug)
        
        # Check if LM Studio SDK is available
        if not HAS_SDK:
            return ("Error: LM Studio SDK is not installed. Please install it using: pip install lmstudio",)

        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting process_image method")
            print(f"Debug: User prompt: {user_prompt}")
            print(f"Debug: Requested Model: {model_key}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")
            print(f"Debug: Image shape: {image.shape}")

        temp_path = None # Initialize temp_path for cleanup

        try:
            # --- Use helper function to get model with fallback ---
            model_obj = get_lm_model_with_fallback(model_key, auto_unload, unload_delay, debug)

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(np.uint8(image[0]*255))

            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                # Save to the temporary file
                pil_image.save(temp_path, format="JPEG")

            if debug:
                print(f"Debug: Saved image to temporary file: {temp_path}")

            # Use the file path method to prepare the image
            image_handle = lms.prepare_image(temp_path)

            # Create a new chat
            chat = lms.Chat(system_prompt)

            # Add user message with image and prompt
            chat.add_user_message(user_prompt, images=[image_handle])

            if debug:
                print(f"Debug: Added image and prompt to chat message")

            # Configure generation parameters
            config = {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "seed": seed
            }

            if debug:
                print(f"Debug: Sending request to LM Studio")
            # --- Timeout logic ---
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model_obj.respond, chat, config=config)
                try:
                    result = future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    error_message = f"Error: LM Studio model response timed out after {timeout_seconds} seconds."
                    print(error_message)
                    return (error_message,)

            if debug:
                print(f"Debug: Response received: {result.content[:100]}...")  # Print first 100 characters
                print(f"Debug: Tokens generated: {result.stats.predicted_tokens_count}")
                print(f"Debug: Time to first token: {result.stats.time_to_first_token_sec}s")

            # Unload model immediately if requested (TTL handles delayed unloading)
            if auto_unload == "True" and unload_delay == 0:
                try:
                    if debug:
                        print(f"Debug: Unloading model immediately.")
                    model_obj.unload()
                except Exception as unload_err:
                    print(f"Warning: Failed to unload model: {unload_err}")

            return (result.content,)

        except Exception as e:
            error_message = f"Error processing with LM Studio: {str(e)}"
            print(error_message)
            return (error_message,)
        finally:
            # Clean up the temporary image file if it was created
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    if debug:
                        print(f"Debug: Removed temporary file: {temp_path}")
                except Exception as cleanup_err:
                    print(f"Warning: Failed to remove temporary file {temp_path}: {cleanup_err}")

    def _process_image_legacy_http(self, image, user_prompt, system_prompt, model, ip_address, port, seed, max_tokens=1000, temperature=0.7, debug=False):
        """Legacy HTTP-based image processing for backward compatibility"""
        print("Warning: Using legacy HTTP mode. Consider upgrading to SDK mode for better performance.")
        
        try:
            import requests
        except ImportError:
            return ("Error: requests library not found. Please install it using: pip install requests",)
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting legacy HTTP process_image method")
            print(f"Debug: Text input: {user_prompt}")
            print(f"Debug: Model: {model}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Image shape: {image.shape}")

        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(np.uint8(image[0]*255))

            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Prepare the payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ]}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "seed": seed
            }

            if debug:
                print(f"Debug: Payload prepared, attempting to connect to server")

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer lm-studio"
            }

            url = f"http://{ip_address}:{port}/v1/chat/completions"
            response = requests.post(url, json=payload, headers=headers, timeout=60)

            if debug:
                print(f"Debug: Server response status code: {response.status_code}")

            response.raise_for_status()

            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                generated_text = response_json['choices'][0]['message']['content']
            else:
                generated_text = "No content in the response"

            if debug:
                print(f"Debug: Generated text: {generated_text[:100]}...")

            return (generated_text,)

        except Exception as e:
            error_message = f"Legacy HTTP Error: {str(e)}"
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
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 3600, "step": 1}),
                # Legacy parameters for backward compatibility
                "model": ("STRING", {"default": ""}),  # Old parameter name
                "ip_address": ("STRING", {"default": ""}),  # Legacy HTTP mode
                "port": ("INT", {"default": 0, "min": 0, "max": 65535}),  # Legacy HTTP mode
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "generate_text"
    CATEGORY = "ComfyExpo/Text"

    @classmethod
    def IS_CHANGED(cls, prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, model="", ip_address="", port=0):
        m = hashlib.sha256()
        
        m.update(str(prompt).encode())
        m.update(str(system_prompt).encode())
        m.update(str(model_key).encode())
        m.update(str(auto_unload).encode())
        m.update(str(unload_delay).encode())
        m.update(str(seed).encode())
        m.update(str(max_tokens).encode())
        m.update(str(temperature).encode())
        m.update(str(debug).encode())
        m.update(str(timeout_seconds).encode())
        m.update(str(model).encode())
        m.update(str(ip_address).encode())
        m.update(str(port).encode())
        
        return m.hexdigest()

    def generate_text(self, prompt, system_prompt, model_key, auto_unload, unload_delay, seed, max_tokens=1000, temperature=0.7, debug=False, timeout_seconds=300, model="", ip_address="", port=0):
        # Handle backward compatibility
        # If legacy parameters are provided, show a deprecation warning and try to use them
        if model and not model_key:
            model_key = model
            if debug:
                print("Debug: Using legacy 'model' parameter as 'model_key'")
        
        if ip_address and port > 0:
            # Legacy HTTP mode detected
            return self._generate_text_legacy_http(prompt, system_prompt, model_key or model, ip_address, port, seed, max_tokens, temperature, debug)
        
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
            print(f"Debug: Requested Model: {model_key}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Auto unload: {auto_unload}, Unload delay: {unload_delay}s")
            print(f"Debug: Max tokens: {max_tokens}")
            print(f"Debug: Temperature: {temperature}")

        try:
            # --- Use helper function to get model with fallback ---
            model_obj = get_lm_model_with_fallback(model_key, auto_unload, unload_delay, debug)

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
            # --- Timeout logic ---
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model_obj.respond, chat, config=config)
                try:
                    result = future.result(timeout=timeout_seconds)
                except concurrent.futures.TimeoutError:
                    error_message = f"Error: LM Studio model response timed out after {timeout_seconds} seconds."
                    print(error_message)
                    return (error_message,)

            if debug:
                print(f"Debug: Response received: {result.content[:100]}...")  # Print first 100 characters
                print(f"Debug: Tokens generated: {result.stats.predicted_tokens_count}")
                print(f"Debug: Time to first token: {result.stats.time_to_first_token_sec}s")

            # Unload model immediately if requested (TTL handles delayed unloading)
            if auto_unload == "True" and unload_delay == 0:
                try:
                    if debug:
                        print(f"Debug: Unloading model immediately.")
                    model_obj.unload()
                except Exception as unload_err:
                    print(f"Warning: Failed to unload model: {unload_err}")

            return (result.content,)

        except Exception as e:
            error_message = f"Error processing with LM Studio: {str(e)}"
            print(error_message)
            return (error_message,)

    def _generate_text_legacy_http(self, prompt, system_prompt, model, ip_address, port, seed, max_tokens=1000, temperature=0.7, debug=False):
        """Legacy HTTP-based text generation for backward compatibility"""
        print("Warning: Using legacy HTTP mode. Consider upgrading to SDK mode for better performance.")
        
        try:
            import requests
        except ImportError:
            return ("Error: requests library not found. Please install it using: pip install requests",)
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting legacy HTTP generate_text method")
            print(f"Debug: Prompt: {prompt}")
            print(f"Debug: Model: {model}")
            print(f"Debug: System prompt: {system_prompt}")

        try:
            # Prepare the payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "seed": seed
            }

            if debug:
                print(f"Debug: Payload prepared, attempting to connect to server")

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer lm-studio"
            }

            url = f"http://{ip_address}:{port}/v1/chat/completions"
            response = requests.post(url, json=payload, headers=headers, timeout=60)

            if debug:
                print(f"Debug: Server response status code: {response.status_code}")

            response.raise_for_status()

            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                generated_text = response_json['choices'][0]['message']['content']
            else:
                generated_text = "No content in the response"

            if debug:
                print(f"Debug: Generated text: {generated_text[:100]}...")

            return (generated_text,)

        except Exception as e:
            error_message = f"Legacy HTTP Error: {str(e)}"
            print(error_message)
            return (error_message,)
