# __init__.py

# Import all node classes from your single code file
from .lmstudio_nodes import (
    ExpoLmstudioUnified,
    ExpoLmstudioImageToText,
    ExpoLmstudioTextGeneration,
    ExpoLmstudioModelManager,
    ExpoLmstudioModelSelector,
    ExpoLmstudioSetup
)

# Define how ComfyUI maps the node name (used in backend) to the class
NODE_CLASS_MAPPINGS = {
    "Expo Lmstudio Unified": ExpoLmstudioUnified,
    "Expo Lmstudio Image To Text": ExpoLmstudioImageToText,
    "Expo Lmstudio Text Generation": ExpoLmstudioTextGeneration,
    "Expo Lmstudio Model Manager": ExpoLmstudioModelManager,
    "Expo Lmstudio Model Selector": ExpoLmstudioModelSelector,
    "Expo Lmstudio Setup": ExpoLmstudioSetup,
    # Use unique internal names if needed, but these should be fine
}

# Define how ComfyUI maps the node name to its display name (shown in the UI)
NODE_DISPLAY_NAME_MAPPINGS = {
    "Expo Lmstudio Unified": "LM Studio Unified (Expo)", # Added (Expo) for clarity
    "Expo Lmstudio Image To Text": "LM Studio I2T (Expo)",
    "Expo Lmstudio Text Generation": "LM Studio Text Gen (Expo)",
    "Expo Lmstudio Model Manager": "LM Studio Model Mgr (Expo)",
    "Expo Lmstudio Model Selector": "LM Studio Model Sel (Expo)",
    "Expo Lmstudio Setup": "LM Studio Setup (Expo)",
    # Make these names descriptive for the UI
}

# Standard dictionary telling ComfyUI what this package provides
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("--- ComfyExpo LM Studio Nodes Loaded ---")