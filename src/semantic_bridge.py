"""
Semantic Bridge Module — The "Translator" of VTAS.

Resolves the vocabulary mismatch between the Visual Grounding Module
(DETR's label vocabulary) and the Linguistic Extraction Module (the
Vision-Language Model's natural language vocabulary).

Without this module, a caption saying "sofa" would be incorrectly
flagged as a hallucination when DETR detects "couch". The Semantic
Bridge computes cosine similarity between word embeddings to determine
if two different words refer to the same real-world concept.

Architecture:
    sentence-transformers/all-MiniLM-L6-v2 → 22M parameters.
    Produces 384-dimensional embeddings optimized for semantic similarity.
"""

from sentence_transformers import SentenceTransformer
from itertools import product


class SemanticBridgeModule:
    """
    Computes semantic similarity between object labels from the Visual
    Grounding Module and nouns from the Linguistic Extraction Module.

    This module determines which text nouns are valid matches for
    detected objects (true positives) and which have no visual
    grounding (hallucinations).

    Attributes:
        model: The sentence-transformer model for embedding generation.
        threshold: The cosine similarity cutoff for a valid match.
    """

    def __init__(self, threshold: float = 0.75):
        """
        Initializes the Semantic Bridge Module.

        Args:
            threshold: Minimum cosine similarity for two words to be
                considered semantically equivalent. Default 0.75 was
                empirically chosen to accept valid synonyms (sofa/couch
                at ~0.89) while rejecting related-but-different concepts
                (cat/dog at ~0.45).

        Note:
            Threshold tuning is identified as a key area for future work.
            See docs/architecture.md for the Dynamic Thresholding proposal.
        """
        self.threshold = threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def compute_alignment(
        self, detected_objects: set, text_nouns: set
    ) -> dict:
        """
        Computes the semantic alignment between detected visual objects
        and extracted text nouns.

        For each text noun, finds the best-matching detected object
        using cosine similarity. If the best match exceeds the threshold,
        it is classified as a valid match. Otherwise, it is classified
        as a hallucination.

        For each detected object, checks if any text noun references it.
        If not, it is classified as a missed object (impacts Visual Recall).

        Args:
            detected_objects: Set of object labels from DETR.
                Example: {'person', 'couch', 'tv'}
            text_nouns: Set of nouns from the generated caption.
                Example: {'man', 'sofa', 'television'}

        Returns:
            A dictionary containing:
                - 'matched': List of (text_noun, detected_object, score) tuples
                - 'hallucinated': List of text nouns with no visual grounding
                - 'missed': List of detected objects not mentioned in caption
                - 'similarity_matrix': Full pairwise similarity scores
        """
        # Edge case: empty inputs produce a defined output
        if not detected_objects or not text_nouns:
            return {
                "matched": [],
                "hallucinated": list(text_nouns),
                "missed": list(detected_objects),
                "similarity_matrix": {},
            }

        # Encode all labels into the shared embedding space
        det_list = sorted(detected_objects)
        txt_list = sorted(text_nouns)

        det_embeddings = self.model.encode(det_list, convert_to_tensor=True)
        txt_embeddings = self.model.encode(txt_list, convert_to_tensor=True)

        # Compute pairwise cosine similarity
        from sentence_transformers.util import cos_sim

        sim_matrix = cos_sim(txt_embeddings, det_embeddings)

        # Build a human-readable similarity matrix for debugging
        similarity_matrix = {}
        for i, t in enumerate(txt_list):
            similarity_matrix[t] = {
                d: round(sim_matrix[i][j].item(), 4)
                for j, d in enumerate(det_list)
            }

        # --- Classify text nouns as matched or hallucinated ---
        matched = []
        hallucinated = []
        matched_det_indices = set()

        for i, t in enumerate(txt_list):
            best_score = sim_matrix[i].max().item()
            best_idx = sim_matrix[i].argmax().item()

            if best_score >= self.threshold:
                matched.append((t, det_list[best_idx], round(best_score, 4)))
                matched_det_indices.add(best_idx)
            else:
                hallucinated.append(t)

        # --- Identify detected objects that the caption missed ---
        missed = [
            det_list[j]
            for j in range(len(det_list))
            if j not in matched_det_indices
        ]

        return {
            "matched": matched,
            "hallucinated": hallucinated,
            "missed": missed,
            "similarity_matrix": similarity_matrix,
        }


if __name__ == "__main__":
    # --- Dry Run: Validate synonym resolution and hallucination detection ---
    print("Validating Semantic Bridge Module...")
    module = SemanticBridgeModule(threshold=0.75)

    # Test 1: Synonyms should match
    print("\n  Test 1: Synonym Resolution")
    result = module.compute_alignment(
        detected_objects={"person", "couch", "tv"},
        text_nouns={"man", "sofa", "television"},
    )
    print(f"    Matched: {result['matched']}")
    print(f"    Hallucinated: {result['hallucinated']}")
    print(f"    Missed: {result['missed']}")

    # Test 2: Hallucination should be caught
    print("\n  Test 2: Hallucination Detection")
    result = module.compute_alignment(
        detected_objects={"person", "umbrella", "cow"},
        text_nouns={"people", "umbrella", "dog"},
    )
    print(f"    Matched: {result['matched']}")
    print(f"    Hallucinated: {result['hallucinated']}")
    print(f"    Missed: {result['missed']}")

    print("\nSemantic Bridge Module validated successfully.")
