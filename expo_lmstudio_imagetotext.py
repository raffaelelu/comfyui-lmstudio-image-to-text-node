"""
@author: Matt John Powell
@title: LM Studio Nodes for ComfyUI
@nickname: LM Studio Nodes
@description: This extension provides two custom nodes for ComfyUI that integrate LM Studio's capabilities:
1. Image to Text: Generates text descriptions of images using vision models.
2. Text Generation: Generates text based on a given prompt using language models.
Both nodes offer flexible and customizable workflows working with LM Studio's local API.
"""


import base64
import requests
import json
import numpy as np
from PIL import Image
import io
import random


class ExpoLmstudioUnified:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "lmstudio-community/gemma-3-4b-it"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "ip_address": ("STRING", {"default": "localhost"}),
                "port": ("INT", {"default": 1234, "min": 1, "max": 65535}),
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

    def process_input(self, model, system_prompt, ip_address, port, seed, image=None, text_input="", max_tokens=1000, temperature=0.7, debug=False):
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
            print(f"Debug: Model: {model}")

        try:
            # Prepare the payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
                "seed": seed
            }
            
            # Add user message content based on inputs
            content = []
            
            # Add text if provided
            if has_text:
                content.append({"type": "text", "text": text_input})
            
            # Add image if provided
            if has_image:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(np.uint8(image[0]*255))
                
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

            # Add the user message with all content
            payload["messages"].append({"role": "user", "content": content})

            if debug:
                print(f"Debug: Payload prepared, attempting to connect to server")

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer lm-studio"
            }

            url = f"http://{ip_address}:{port}/v1/chat/completions"
            response = requests.post(url, 
                                     json=payload, 
                                     headers=headers, 
                                     timeout=60)
            
            if debug:
                print(f"Debug: Server response status code: {response.status_code}")

            response.raise_for_status()  # Raise an exception for bad status codes

            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                result_text = response_json['choices'][0]['message']['content']
            else:
                result_text = "No content in the response"

            if debug:
                print(f"Debug: Response received: {result_text[:100]}...")  # Print first 100 characters

            return (result_text,)

        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with the server: {str(e)}"
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            print(error_message)
            return (error_message,)

class ExpoLmstudioImageToText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "lmstudio-community/gemma-3-4b-it"}),
                "system_prompt": ("STRING", {"default": "This is a chat between a user and an assistant. The assistant is an expert in describing images, with detail and accuracy"}),
                "ip_address": ("STRING", {"default": "localhost"}),
                "port": ("INT", {"default": 1234, "min": 1, "max": 65535}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "user_prompt": ("STRING", {"default": "Describe this image in detail"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Description",)
    FUNCTION = "process_image"
    CATEGORY = "ComfyExpo/I2T"

    def process_image(self, model, system_prompt, ip_address, port, seed, image=None, user_prompt="Describe this image in detail", debug=False):
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
            print(f"Debug: Model: {model}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: IP Address: {ip_address}")
            print(f"Debug: Port: {port}")
            if image is not None:
                print(f"Debug: Image shape: {image.shape}")

        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(np.uint8(image[0]*255))
            
            # Convert PIL Image to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            if debug:
                print(f"Debug: Image successfully converted to base64")

            # Prepare the payload for the server
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                "max_tokens": 1000,
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
            response = requests.post(url, 
                                     json=payload, 
                                     headers=headers, 
                                     timeout=60)
            
            if debug:
                print(f"Debug: Server response status code: {response.status_code}")

            response.raise_for_status()  # Raise an exception for bad status codes

            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                response_text = response_json['choices'][0]['message']['content']
            else:
                response_text = "No content in the response"

            if debug:
                print(f"Debug: Response received: {response_text[:100]}...")  # Print first 100 characters

            return (response_text,)

        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with the server: {str(e)}"
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            print(error_message)
            return (error_message,)
        
class ExpoLmstudioTextGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "lmstudio-community/gemma-3-4b-it"}),
                "system_prompt": ("STRING", {"default": "You are a helpful AI assistant."}),
                "ip_address": ("STRING", {"default": "localhost"}),
                "port": ("INT", {"default": 1234, "min": 1, "max": 65535}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "Generate a creative story:"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Generated Text",)
    FUNCTION = "generate_text"
    CATEGORY = "ComfyExpo/Text"

    def generate_text(self, model, system_prompt, ip_address, port, seed, prompt="Generate a creative story:", max_tokens=1000, temperature=0.7, debug=False):
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
            print(f"Debug: Model: {model}")
            print(f"Debug: System prompt: {system_prompt}")
            print(f"Debug: IP Address: {ip_address}")
            print(f"Debug: Port: {port}")
            print(f"Debug: Max tokens: {max_tokens}")
            print(f"Debug: Temperature: {temperature}")

        try:
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
            response = requests.post(url, 
                                     json=payload, 
                                     headers=headers, 
                                     timeout=60)
            
            if debug:
                print(f"Debug: Server response status code: {response.status_code}")

            response.raise_for_status()  # Raise an exception for bad status codes

            response_json = response.json()
            if 'choices' in response_json and len(response_json['choices']) > 0:
                generated_text = response_json['choices'][0]['message']['content']
            else:
                generated_text = "No content in the response"

            if debug:
                print(f"Debug: Generated text: {generated_text[:100]}...")  # Print first 100 characters

            return (generated_text,)

        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with the server: {str(e)}"
            print(error_message)
            return (error_message,)
        except Exception as e:
            error_message = f"Unexpected error: {str(e)}"
            print(error_message)
            return (error_message,)