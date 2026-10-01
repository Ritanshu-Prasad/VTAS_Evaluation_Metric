"""
Linguistic Extraction Module — The "Brain" of VTAS.

Parses a generated caption string and extracts the set of primary
nouns (objects) that the Vision-Language Model claims are present
in the image. Uses spaCy's pre-trained English NLP pipeline for
Part-of-Speech (POS) tagging.

Design Decision:
    We extract only nouns (POS tags: NOUN, PROPN) because VTAS v1
    evaluates object-level alignment. Adjectives, verbs, and spatial
    prepositions are out-of-scope for v1 and are planned for v1.1
    (Attribute-Level Grounding and Spatial Awareness).
"""

import re
import spacy


class LinguisticExtractionModule:
    """
    Extracts a set of noun objects from a natural language caption
    using spaCy's Part-of-Speech tagger. Includes a preprocessing
    stage that strips common prompt-engineering artifacts before
    noun extraction.

    Attributes:
        nlp: The spaCy English language model.
        STOP_NOUNS: A set of generic nouns to filter out.
        CAPTION_PREFIXES: Common prompt artifacts to strip from input.
    """

    # Common prompt-engineering prefixes that VLMs echo back verbatim.
    # These are stripped before NLP parsing to prevent words like
    # "caption" or "description" from being extracted as nouns.
    CAPTION_PREFIXES = [
        "detailed caption",
        "descriptive caption",
        "short caption",
        "brief caption",
        "long caption",
        "caption",
        "detailed description",
        "short description",
        "brief description",
        "description",
    ]

    # Generic nouns that appear frequently in captions but do not
    # represent identifiable visual objects in an image. Organized
    # into semantic categories for maintainability.
    STOP_NOUNS = {
        # --- Quantifiers and groupings ---
        "group", "bunch", "couple", "pair", "lot", "number",
        "collection", "series", "row", "stack", "pile",
        # --- Meta-references to the image itself ---
        "photo", "picture", "image", "scene", "view", "area",
        "shot", "frame", "photograph", "video", "clip",
        "caption", "description", "text",
        # --- Spatial/positional terms ---
        "side", "top", "bottom", "front", "back", "middle",
        "center", "edge", "corner", "end", "section",
        # --- Abstract/vague nouns ---
        "way", "kind", "type", "set", "bit", "part",
        "thing", "stuff", "piece", "item", "object",
        # --- Environment/scene descriptors (DETR cannot detect these) ---
        "room", "bedroom", "kitchen", "bathroom", "living",
        "outside", "inside", "outdoors", "indoors", "exterior",
        "interior", "hallway", "corridor", "lobby", "garage",
        "garden", "yard", "patio", "balcony", "terrace",
        "street", "road", "highway", "alley", "sidewalk",
        "beach", "ocean", "lake", "river", "mountain",
        "forest", "park", "field", "city", "town",
        # --- Atmospheric/lighting conditions ---
        "color", "colour", "background", "foreground",
        "sunlight", "shadow", "darkness", "light", "sky",
        "weather", "rain", "snow", "fog", "night", "day",
    }

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initializes the Linguistic Extraction Module.

        Args:
            model_name: The spaCy model to load. 'en_core_web_sm' is
                the lightweight default suitable for POS tagging.
                For production, 'en_core_web_md' or 'en_core_web_lg'
                offer better accuracy at higher memory cost.
        """
        self.nlp = spacy.load(model_name)

    def _strip_prefix(self, caption: str) -> str:
        """
        Removes common prompt-engineering artifacts from the start
        of a generated caption. This handles cases where VLMs echo
        back the instruction prefix (e.g., "detailed caption the dog
        is sitting" -> "the dog is sitting").

        Uses longest-match-first ordering to prevent partial stripping
        (e.g., "detailed caption" is checked before "caption").

        Args:
            caption: The raw generated caption string.

        Returns:
            The caption with any matching prefix removed and whitespace
            normalized.
        """
        lower = caption.lower().strip()

        for prefix in self.CAPTION_PREFIXES:
            if lower.startswith(prefix):
                caption = caption[len(prefix):].strip()
                # Strip optional separators like ":" or "-" after prefix
                caption = re.sub(r'^[\s:\-–—]+', '', caption).strip()
                break

        return caption

    def extract(self, caption: str) -> set:
        """
        Parses the input caption and returns a set of unique nouns
        representing the objects the model claims are in the image.

        Processing pipeline:
            1. Strip prompt-engineering prefixes.
            2. Lowercase and tokenize with spaCy.
            3. Extract NOUN and PROPN tokens.
            4. Lemmatize to normalize plurals ("dogs" -> "dog").
            5. Filter out stop nouns.

        Args:
            caption: The generated caption string.
                Example: "detailed caption A man throws a frisbee"

        Returns:
            A set of lowercase noun strings, filtered and cleaned.
            Example: {'man', 'frisbee'}
        """
        # Stage 1: Strip prompt artifacts
        cleaned = self._strip_prefix(caption)

        # Stage 2: NLP parsing
        doc = self.nlp(cleaned.lower())

        extracted_nouns = set()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"):
                lemma = token.lemma_
                # Filter stop nouns and single-character artifacts
                if lemma not in self.STOP_NOUNS and len(lemma) > 1:
                    extracted_nouns.add(lemma)

        return extracted_nouns


if __name__ == "__main__":
    # --- Dry Run: Validate prefix stripping and noun extraction ---
    print("Validating Linguistic Extraction Module...")
    module = LinguisticExtractionModule()

    test_cases = [
        "detailed caption the living room has a yellow wall and a black fireplace",
        "short caption a man throws a frisbee to his dog in the park",
        "A group of people standing with umbrellas and dogs",
        "descriptive caption: A woman sits on a red couch watching television",
        "The bedroom has a large bed and a window",
    ]

    for caption in test_cases:
        nouns = module.extract(caption)
        print(f"  Input:    \"{caption}\"")
        print(f"  Cleaned:  \"{module._strip_prefix(caption)}\"")
        print(f"  Nouns:    {nouns}\n")

    print("Linguistic Extraction Module validated successfully.")
