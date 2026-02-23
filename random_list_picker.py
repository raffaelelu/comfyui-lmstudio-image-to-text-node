"""
Random List Picker Node for ComfyUI
Paste a newline-separated list of items; the node picks one at random and
outputs it as a string along with the total count of items in the list.
An optional prefix and suffix let you wrap the chosen item inside a full
prompt, e.g. prefix="Create a song using this genre: " + chosen + suffix.
"""

import random


class RandomListPicker:
    """
    Accepts a multiline string where each non-empty line is treated as one item.
    Randomly selects one item and returns it together with the total item count
    and an optional fully-assembled prompt string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "rock\npop\njazz\nclassical\nhip-hop",
                        "tooltip": "Paste one item per line. Empty lines are ignored.",
                    },
                ),
            },
            "optional": {
                "prefix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Text to prepend before the chosen item in the prompt output.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Text to append after the chosen item in the prompt output.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0x7FFFFFFF,
                        "tooltip": "Random seed. Use -1 for a truly random pick each run.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("selected_item", "prompt", "item_count")
    FUNCTION = "pick_random"
    CATEGORY = "utils"

    def pick_random(self, items: str, prefix: str = "", suffix: str = "", seed: int = -1):
        # Split on newlines and strip whitespace; discard empty lines
        lines = [line.strip() for line in items.splitlines()]
        clean = [line for line in lines if line]

        if not clean:
            return ("", "", 0)

        if seed != -1:
            rng = random.Random(seed)
            chosen = rng.choice(clean)
        else:
            chosen = random.choice(clean)

        prompt = f"{prefix}{chosen}{suffix}"

        return (chosen, prompt, len(clean))
