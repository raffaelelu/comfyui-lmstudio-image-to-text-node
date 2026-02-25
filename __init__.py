"""
LM Studio Nodes for ComfyUI - Integration with local LLM models
"""

# Import all node classes from the main module file
from .expo_lmstudio_imagetotext import (
    ExpoLmstudioUnified,
    ExpoLmstudioImageToText,
    ExpoLmstudioTextGeneration,
    ExpoLmstudioStructuredOutput
)
from .random_list_picker import RandomListPicker

# Define how ComfyUI maps the node name (used in backend) to the class
NODE_CLASS_MAPPINGS = {
    "Expo Lmstudio Unified": ExpoLmstudioUnified,
    "Expo Lmstudio Image To Text": ExpoLmstudioImageToText,
    "Expo Lmstudio Text Generation": ExpoLmstudioTextGeneration,
    "Expo Lmstudio Structured Output": ExpoLmstudioStructuredOutput,
    "Random List Picker": RandomListPicker
}

# Define how ComfyUI maps the node name to its display name (shown in the UI)
NODE_DISPLAY_NAME_MAPPINGS = {
    "Expo Lmstudio Unified": "LM Studio (Unified)",
    "Expo Lmstudio Image To Text": "LM Studio (Image to Text)",
    "Expo Lmstudio Text Generation": "LM Studio (Text Gen)",
    "Expo Lmstudio Structured Output": "LM Studio (Structured Output)",
    "Random List Picker": "Random List Picker"
}

# Standard dictionary telling ComfyUI what this package provides
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("--- ComfyExpo LM Studio Nodes Loaded ---")
