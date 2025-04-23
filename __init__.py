"""
LM Studio Nodes for ComfyUI - Integration with local LLM models
"""

# Import all node classes from the main module file
from .expo_lmstudio_imagetotext import (
    ExpoLmstudioUnified,
    ExpoLmstudioImageToText,
    ExpoLmstudioTextGeneration,
    ExpoLmstudioSetup,
    refresh_available_models
)

# Try to load models at initialization time
try:
    refresh_available_models()
except Exception as e:
    print(f"Note: Could not pre-load LM Studio models. They will be loaded when needed: {str(e)}")

# Define how ComfyUI maps the node name (used in backend) to the class
NODE_CLASS_MAPPINGS = {
    "Expo Lmstudio Unified": ExpoLmstudioUnified,
    "Expo Lmstudio Image To Text": ExpoLmstudioImageToText,
    "Expo Lmstudio Text Generation": ExpoLmstudioTextGeneration,
    "Expo Lmstudio Setup": ExpoLmstudioSetup
}

# Define how ComfyUI maps the node name to its display name (shown in the UI)
NODE_DISPLAY_NAME_MAPPINGS = {
    "Expo Lmstudio Unified": "LM Studio (Unified)",
    "Expo Lmstudio Image To Text": "LM Studio (Image to Text)",
    "Expo Lmstudio Text Generation": "LM Studio (Text Gen)",
    "Expo Lmstudio Setup": "LM Studio (Setup)"
}

# Standard dictionary telling ComfyUI what this package provides
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("--- ComfyExpo LM Studio Nodes Loaded ---")
