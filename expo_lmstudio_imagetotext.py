"""
@author: Matt John Powell
@title: LM Studio Nodes for ComfyUI
@nickname: LM Studio Nodes
@description: This extension provides three custom nodes for ComfyUI that integrate LM Studio's capabilities:
1. Image to Text: Generates text descriptions of images using vision models.
2. Text Generation: Generates text based on a given prompt using language models.
3. Model Manager: Lists, loads, and unloads models to manage memory usage.
All nodes leverage the official LM Studio Python SDK for better performance and reliability.
"""

import base64
import numpy as np
from PIL import Image
import io
import random
import lmstudio as lms

class ExpoLmstudioUnified:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_key": ("STRING", {"default": "llama-3.2-1b-instruct"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "text_input": ("STRING", {"default": ""}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "process_input"
    CATEGORY = "ComfyExpo/LMStudio"

    def process_input(self, model_key, system_prompt, seed, image=None, text_input="", max_tokens=1000, temperature=0.7, debug=False):
        # Check if we have valid inputs
        has_image = image is not None
        has_text = text_input.strip() != ""
        
        # If no inputs are provided, disable the node
        if not has_image and not has_text:
            print("No inputs provided. The node will not run.")
            return ("No inputs provided. Please connect an image or provide text input.",)
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)
        
        if debug:
            print(f"Debug: Starting process_input method")
            print(f"Debug: Has image input: {has_image}")
            print(f"Debug: Has text input: {has_text}")
            print(f"Debug: Model: {model_key}")

        try:
            # Get model reference
            model = lms.llm(model_key)
            
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
                "model_key": ("STRING", {"default": "qwen2-vl-2b-instruct"}),
                "system_prompt": ("STRING", {"default": "This is a chat between a user and an assistant. The assistant is an expert in describing images, with detail and accuracy"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "Describe this image in detail"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)
    FUNCTION = "process_image"
    CATEGORY = "ComfyExpo/I2T"

    def process_image(self, model_key, system_prompt, seed, image=None, user_prompt="Describe this image in detail", max_tokens=1000, temperature=0.7, debug=False):
        # Check if we have valid inputs
        if image is None:
            print("No image provided. The node will not run.")
            return ("No image provided. Please connect an image input.",)
            
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)
        
        if debug:
            print(f"Debug: Starting process_image method")
            print(f"Debug: Text input: {user_prompt}")
            print(f"Debug: Model: {model_key}")
            print(f"Debug: System prompt: {system_prompt}")
            if image is not None:
                print(f"Debug: Image shape: {image.shape}")

        try:
            # Get model reference - make sure it's a vision-language model
            model = lms.llm(model_key)
            
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
            result = model.respond(chat, config=config)
            
            if debug:
                print(f"Debug: Response received: {result.content[:100]}...")  # Print first 100 characters
                print(f"Debug: Tokens generated: {result.stats.predicted_tokens_count}")
                print(f"Debug: Generation time: {result.stats.generation_time_sec}s")
            
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
                "model_key": ("STRING", {"default": "llama-3.2-1b-instruct"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "Generate a creative story:"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "stream_output": ("BOOLEAN", {"default": False}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "generate_text"
    CATEGORY = "ComfyExpo/Text"

    def generate_text(self, model_key, system_prompt, seed, prompt="Generate a creative story:", max_tokens=1000, temperature=0.7, stream_output=False, debug=False):
        # Check if we have valid inputs
        if prompt.strip() == "":
            print("No prompt provided. The node will not run.")
            return ("No prompt provided. Please provide a text prompt.",)
            
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        random.seed(seed)

        if debug:
            print(f"Debug: Starting generate_text method")
            print(f"Debug: Prompt: {prompt}")
            print(f"Debug: Model: {model_key}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: Max tokens: {max_tokens}")
            print(f"Debug: Temperature: {temperature}")
            print(f"Debug: Stream output: {stream_output}")

        try:
            # Get model reference
            model = lms.llm(model_key)
            
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
            
            # Generate response (with streaming option)
            if stream_output:
                # Stream response (might not be directly displayable in ComfyUI)
                full_text = ""
                for fragment in model.respond_stream(chat, config=config):
                    if debug:
                        print(f"Debug: Fragment received: {fragment.content}")
                    full_text += fragment.content
                
                if debug:
                    print(f"Debug: Full streamed text: {full_text[:100]}...")
                
                return (full_text,)
            else:
                # Standard response
                result = model.respond(chat, config=config)
                
                if debug:
                    print(f"Debug: Response received: {result.content[:100]}...")
                    print(f"Debug: Tokens generated: {result.stats.predicted_tokens_count}")
                    print(f"Debug: Generation time: {result.stats.generation_time_sec}s")
                
                return (result.content,)

        except Exception as e:
            error_message = f"Error processing with LM Studio: {str(e)}"
            print(error_message)
            return (error_message,)

class ExpoLmstudioModelManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["LIST", "LOAD", "UNLOAD"], {"default": "LIST"}),
            },
            "optional": {
                "model_key": ("STRING", {"default": ""}),
                "model_type": (["ALL", "LLM", "EMBEDDING"], {"default": "ALL"}),
                "load_ttl": ("INT", {"default": 3600, "min": 0, "max": 86400}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "manage_models"
    CATEGORY = "ComfyExpo/LMStudio"

    def manage_models(self, action, model_key="", model_type="ALL", load_ttl=3600, debug=False):
        if debug:
            print(f"Debug: Starting manage_models method")
            print(f"Debug: Action: {action}")
            print(f"Debug: Model key: {model_key}")
            print(f"Debug: Model type: {model_type}")
            print(f"Debug: Load TTL: {load_ttl}")
        
        try:
            result = ""
            
            if action == "LIST":
                # List downloaded models of the specified type
                if model_type == "LLM":
                    models = lms.list_downloaded_models("llm")
                    result = "Available LLM Models:\n"
                elif model_type == "EMBEDDING":
                    models = lms.list_downloaded_models("embedding")
                    result = "Available Embedding Models:\n"
                else:  # ALL
                    models = lms.list_downloaded_models()
                    result = "All Available Models:\n"
                
                # Format the model list
                for model in models:
                    result += f"- {model.display_name} (Key: {model.model_key})"
                    if hasattr(model, 'vision') and model.vision:
                        result += " [Vision capable]"
                    result += "\n"
                
                if debug:
                    print(f"Debug: Found {len(models)} models")
            
            elif action == "LOAD":
                if not model_key:
                    return ("Error: No model key provided for LOAD action.",)
                
                if model_type == "EMBEDDING":
                    # Load embedding model
                    model = lms.embedding_model(model_key, ttl=load_ttl)
                    result = f"Successfully loaded embedding model: {model_key} with TTL: {load_ttl}s"
                else:
                    # Default to LLM
                    model = lms.llm(model_key, ttl=load_ttl)
                    result = f"Successfully loaded LLM model: {model_key} with TTL: {load_ttl}s"
                
                if debug:
                    print(f"Debug: Loaded model: {model_key}")
            
            elif action == "UNLOAD":
                if not model_key:
                    return ("Error: No model key provided for UNLOAD action.",)
                
                # Get the model and unload it
                if model_type == "EMBEDDING":
                    model = lms.embedding_model(model_key)
                    model.unload()
                    result = f"Successfully unloaded embedding model: {model_key}"
                else:
                    # Default to LLM
                    model = lms.llm(model_key)
                    model.unload()
                    result = f"Successfully unloaded LLM model: {model_key}"
                
                if debug:
                    print(f"Debug: Unloaded model: {model_key}")
            
            return (result,)
        
        except Exception as e:
            error_message = f"Error managing models: {str(e)}"
            print(error_message)
            return (error_message,)

# Additional UI node to provide model listings as dropdown options
class ExpoLmstudioModelSelector:
    @classmethod
    def INPUT_TYPES(cls):
        # Try to get the list of models from LM Studio
        try:
            models = lms.list_downloaded_models("llm")
            model_keys = [model.model_key for model in models]
            
            if not model_keys:
                model_keys = ["llama-3.2-1b-instruct"]  # Default fallback
                
            # Get all vision-capable models
            vision_models = [model.model_key for model in models if hasattr(model, 'vision') and model.vision]
            
            # No vision models found, add a default example
            if not vision_models:
                vision_models = ["qwen2-vl-2b-instruct"]  # Default vision model example
                
            return {
                "required": {
                    "model_type": (["LLM", "Vision"], {"default": "LLM"}),
                },
                "optional": {
                    "filter_text": ("STRING", {"default": ""}),
                }
            }
        except Exception as e:
            print(f"Error loading model list: {str(e)}")
            return {
                "required": {
                    "model_type": (["LLM", "Vision"], {"default": "LLM"}),
                },
                "optional": {
                    "filter_text": ("STRING", {"default": ""}),
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Model Key",)
    FUNCTION = "select_model"
    CATEGORY = "ComfyExpo/LMStudio"
    
    def select_model(self, model_type, filter_text=""):
        try:
            # Get all models
            models = lms.list_downloaded_models("llm")
            
            # Filter by model type
            if model_type == "Vision":
                filtered_models = [model for model in models if hasattr(model, 'vision') and model.vision]
            else:
                filtered_models = models
            
            # Further filter by text if provided
            if filter_text:
                filtered_models = [
                    model for model in filtered_models 
                    if filter_text.lower() in model.model_key.lower() or 
                       filter_text.lower() in model.display_name.lower()
                ]
            
            # Sort by name
            filtered_models.sort(key=lambda x: x.display_name)
            
            # Return the selected model
            if filtered_models:
                selected_model = filtered_models[0].model_key
                return (selected_model,)
            else:
                # Return default model based on type
                if model_type == "Vision":
                    return ("qwen2-vl-2b-instruct",)
                else:
                    return ("llama-3.2-1b-instruct",)
        
        except Exception as e:
            error_message = f"Error selecting model: {str(e)}"
            print(error_message)
            # Return default model based on type
            if model_type == "Vision":
                return ("qwen2-vl-2b-instruct",)
            else:
                return ("llama-3.2-1b-instruct",)

# Node for executing simple model setup scripts
class ExpoLmstudioSetup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["INSTALL SDK", "GET MODEL", "LIST MODELS"], {"default": "LIST MODELS"}),
                "model_key": ("STRING", {"default": "llama-3.2-1b-instruct"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Result",)
    FUNCTION = "setup_action"
    CATEGORY = "ComfyExpo/LMStudio"
    
    def setup_action(self, action, model_key):
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
                else:
                    result = f"Error installing LM Studio SDK:\n{stderr}"
            except Exception as e:
                result = f"Error: {str(e)}"
        
        elif action == "GET MODEL":
            try:
                # This would normally be done via CLI, but we simulate it here
                result = f"To download the model '{model_key}', run this command in your terminal:\n\nlms get {model_key}\n\nThis will download the model to your LM Studio model directory."
            except Exception as e:
                result = f"Error: {str(e)}"
        
        elif action == "LIST MODELS":
            try:
                # Get models via SDK
                models = lms.list_downloaded_models()
                
                result = "Downloaded Models:\n"
                for model in models:
                    result += f"- {model.display_name} (Key: {model.model_key})"
                    if hasattr(model, 'vision') and model.vision:
                        result += " [Vision capable]"
                    result += "\n"
            except Exception as e:
                result = f"Error listing models: {str(e)}"
        
        return (result,)