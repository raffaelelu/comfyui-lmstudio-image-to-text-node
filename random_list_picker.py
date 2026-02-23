"""
Random List Picker Node for ComfyUI

Features:
  - Paste a newline-separated list; pick one or many random items each run.
  - Weighted picking: append ::weight to any line  (e.g. "jazz::3")
  - Shuffle mode: return the entire list in a random order instead of picking.
  - Template injection: use {item} anywhere in a template string.
  - Prefix / suffix: simple wrapper when no template is needed.
  - Case control: original | uppercase | lowercase | title | sentence.
  - Separator: choose how multiple picks are joined (", ", " | ", newline …).
  - Exclude list: block-list of items that should never be chosen.
  - Fallback: value returned when the usable list is empty.
  - Count: draw N unique items in one pass.
  - Index output: zero-based index of the first chosen item.
  - Seed: -1 = fresh random each run; any other value = reproducible.
"""

import random


class RandomListPicker:

    CASE_OPTIONS = ["original", "uppercase", "lowercase", "title", "sentence"]
    MODE_OPTIONS = ["pick", "shuffle"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "rock\npop\njazz::2\nclassical\nhip-hop",
                        "tooltip": (
                            "One item per line. Empty lines are ignored.\n"
                            "Weighted syntax: item::weight  (e.g. jazz::3 is 3× more likely). "
                            "Default weight is 1."
                        ),
                    },
                ),
            },
            "optional": {
                "template": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Prompt template with {item} as a placeholder.\n"
                            "Example: 'Create a {item} song with heavy bass.'\n"
                            "When count > 1, items are joined with the separator before substitution.\n"
                            "Overrides prefix/suffix when non-empty."
                        ),
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Text prepended before the chosen item(s). Used when template is empty.",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Text appended after the chosen item(s). Used when template is empty.",
                    },
                ),
                "exclude": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Items to never pick, one per line. Matching is case-insensitive.",
                    },
                ),
                "fallback": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Value returned when no items remain after exclusions.",
                    },
                ),
                "count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 100,
                        "tooltip": "Number of unique items to pick. Automatically capped at list size.",
                    },
                ),
                "separator": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": ", ",
                        "tooltip": "String used to join multiple picked items (e.g. ', '  ' | '  '\\n').",
                    },
                ),
                "case": (
                    cls.CASE_OPTIONS,
                    {
                        "default": "original",
                        "tooltip": "Case transformation applied to each picked item.",
                    },
                ),
                "mode": (
                    cls.MODE_OPTIONS,
                    {
                        "default": "pick",
                        "tooltip": (
                            "pick   – choose random item(s) from the list.\n"
                            "shuffle – return the entire list in a random order."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 0x7FFFFFFF,
                        "tooltip": "Random seed. Use -1 for a new random result each run.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("selected_item", "prompt", "item_count", "selected_index")
    FUNCTION = "pick_random"
    CATEGORY = "utils"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_case(text: str, case: str) -> str:
        if case == "uppercase":
            return text.upper()
        if case == "lowercase":
            return text.lower()
        if case == "title":
            return text.title()
        if case == "sentence":
            return text.capitalize()
        return text  # "original"

    @staticmethod
    def _parse_items(raw: str):
        """Return list of (label, weight) tuples from a multiline string."""
        result = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if "::" in line:
                label, _, raw_weight = line.rpartition("::")
                label = label.strip()
                try:
                    weight = float(raw_weight.strip())
                    if weight <= 0:
                        weight = 1.0
                except ValueError:
                    weight = 1.0
            else:
                label = line
                weight = 1.0
            result.append((label, weight))
        return result

    @staticmethod
    def _build_prompt(template: str, prefix: str, suffix: str, joined: str) -> str:
        if template.strip():
            return template.replace("{item}", joined)
        return f"{prefix}{joined}{suffix}"

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def pick_random(
        self,
        items: str,
        template: str = "",
        prefix: str = "",
        suffix: str = "",
        exclude: str = "",
        fallback: str = "",
        count: int = 1,
        separator: str = ", ",
        case: str = "original",
        mode: str = "pick",
        seed: int = -1,
    ):
        parsed = self._parse_items(items)

        # Apply exclude list (case-insensitive)
        if exclude.strip():
            exclude_set = {
                line.strip().lower()
                for line in exclude.splitlines()
                if line.strip()
            }
            parsed = [(lbl, w) for lbl, w in parsed if lbl.lower() not in exclude_set]

        total = len(parsed)

        if total == 0:
            return (fallback, fallback, 0, -1)

        rng = random.Random(seed) if seed != -1 else random.Random()

        # ---- shuffle mode: return the whole list in random order ----
        if mode == "shuffle":
            labels = [self._apply_case(lbl, case) for lbl, _ in parsed]
            rng.shuffle(labels)
            joined = separator.join(labels)
            prompt = self._build_prompt(template, prefix, suffix, joined)
            return (joined, prompt, total, -1)

        # ---- pick mode ----
        effective_count = min(count, total)
        all_labels  = [lbl for lbl, _ in parsed]
        all_weights = [w   for _, w  in parsed]

        if effective_count == 1:
            # Weighted single pick
            chosen_label = rng.choices(all_labels, weights=all_weights, k=1)[0]
            chosen_index = all_labels.index(chosen_label)
            joined = self._apply_case(chosen_label, case)
        else:
            # Weighted unique multi-pick (without replacement)
            rem_labels  = list(all_labels)
            rem_weights = list(all_weights)
            chosen_labels = []
            for _ in range(effective_count):
                pick = rng.choices(rem_labels, weights=rem_weights, k=1)[0]
                idx  = rem_labels.index(pick)
                chosen_labels.append(pick)
                rem_labels.pop(idx)
                rem_weights.pop(idx)
            chosen_index  = all_labels.index(chosen_labels[0])
            joined = separator.join(
                self._apply_case(lbl, case) for lbl in chosen_labels
            )

        prompt = self._build_prompt(template, prefix, suffix, joined)
        return (joined, prompt, total, chosen_index)
