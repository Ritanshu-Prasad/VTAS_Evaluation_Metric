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

import spacy


class LinguisticExtractionModule:
    """
    Extracts a set of noun objects from a natural language caption
    using spaCy's Part-of-Speech tagger.

    Attributes:
        nlp: The spaCy English language model.
        stop_nouns: A set of generic nouns to filter out. These are
            common words that do not represent specific visual objects
            and would introduce noise into the scoring engine.
    """

    # Generic nouns that appear frequently in captions but do not
    # represent identifiable visual objects in an image.
    STOP_NOUNS = {
        "group", "bunch", "couple", "pair", "lot", "number",
        "photo", "picture", "image", "scene", "view", "area",
        "side", "top", "bottom", "front", "back", "middle",
        "way", "kind", "type", "set", "bit", "part",
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

    def extract(self, caption: str) -> set:
        """
        Parses the input caption and returns a set of unique nouns
        representing the objects the model claims are in the image.

        Args:
            caption: The generated caption string.
                Example: "A man throws a frisbee to his cat"

        Returns:
            A set of lowercase noun strings, filtered of stop nouns.
            Example: {'man', 'frisbee', 'cat'}
        """
        doc = self.nlp(caption.lower())

        extracted_nouns = set()
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN") and token.text not in self.STOP_NOUNS:
                # Use the lemma (base form) to normalize plurals:
                # "dogs" → "dog", "umbrellas" → "umbrella"
                extracted_nouns.add(token.lemma_)

        return extracted_nouns


if __name__ == "__main__":
    # --- Dry Run: Validate the module extracts nouns correctly ---
    print("Validating Linguistic Extraction Module...")
    module = LinguisticExtractionModule()

    test_cases = [
        "A man throws a frisbee to his dog in the park",
        "A group of people standing with umbrellas and dogs",
        "A woman sits on a red couch watching television",
    ]

    for caption in test_cases:
        nouns = module.extract(caption)
        print(f"  Caption: \"{caption}\"")
        print(f"  Extracted Nouns: {nouns}\n")

    print("Linguistic Extraction Module validated successfully.")
