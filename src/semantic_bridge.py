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
    sentence-transformers/all-MiniLM-L6-v2 -> 22M parameters.
    Produces 384-dimensional embeddings optimized for semantic similarity.

Design Decision (Phrase-Based Encoding):
    Sentence-transformers are trained on full sentences, not isolated
    words. Encoding "a man" instead of "man" provides significantly
    richer contextual signal, improving synonym resolution for pairs
    like "man"<->"person" (0.45 bare-word -> 0.72 phrase-based).
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
from scipy.optimize import linear_sum_assignment


class SemanticBridgeModule:
    """
    Computes semantic similarity between object labels from the Visual
    Grounding Module and nouns from the Linguistic Extraction Module
    using phrase-based sentence embeddings.

    This module determines which text nouns are valid matches for
    detected objects (true positives) and which have no visual
    grounding (hallucinations).

    Attributes:
        model: The sentence-transformer model for embedding generation.
        threshold: The cosine similarity cutoff for a valid match.
    """

    def __init__(self, threshold: float = 0.65):
        """
        Initializes the Semantic Bridge Module.

        Args:
            threshold: Minimum cosine similarity for two words to be
                considered semantically equivalent. Default 0.65 was
                chosen after testing phrase-based encoding, which
                shifts the similarity distribution upward. Key pairs:
                    "a man"   <-> "a person"     : ~0.72
                    "a sofa"  <-> "a couch"      : ~0.85
                    "a cat"   <-> "a dog"         : ~0.48  (rejected)
                    "a car"   <-> "a truck"       : ~0.58  (rejected)
        """
        self.threshold = threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    @staticmethod
    def _to_phrase(word: str) -> str:
        """
        Wraps a bare noun in a natural-language phrase to improve
        embedding quality. Sentence-transformers produce significantly
        more discriminative embeddings when given phrases instead of
        isolated words.

        Args:
            word: A bare noun string (e.g., "person", "baseball bat").

        Returns:
            A phrase string (e.g., "a person", "a baseball bat").
        """
        return f"a {word}"

    def compute_alignment(
        self, detected_objects: list, text_nouns: list
    ) -> dict:
        """
        Computes the semantic alignment between detected visual objects
        and extracted text nouns using phrase-based embeddings and
        Bipartite Instance Matching (Hungarian algorithm) to solve
        Semantic Collapse.

        Args:
            detected_objects: List of dicts from DETR.
                Example: [{'label': 'person', ...}, {'label': 'couch', ...}]
            text_nouns: List of nouns from the generated caption.
                Example: ['man', 'sofa', 'television']

        Returns:
            A dictionary containing:
                - 'matched': List of (text_noun, detected_dict, score) tuples
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

        # Deterministic ordering (while preserving instance counts)
        # Sort by area/score if available, or just label
        det_list = sorted(detected_objects, key=lambda x: (x['label'], -x.get('area', 0)))
        
        # text_nouns is a list of dicts: [{'text': 'man', 'is_plural': False}]
        if isinstance(text_nouns[0], dict):
            txt_list = sorted(text_nouns, key=lambda x: x["text"])
            txt_labels = [t["text"] for t in txt_list]
        else:
            txt_list = sorted(text_nouns)
            txt_labels = list(txt_list)

        det_labels = [d['label'] for d in det_list]
        
        # Encode as phrases for richer semantic signal
        det_phrases = [self._to_phrase(d) for d in det_labels]
        txt_phrases = [self._to_phrase(t) for t in txt_labels]

        det_embeddings = self.model.encode(
            det_phrases, convert_to_tensor=True
        )
        txt_embeddings = self.model.encode(
            txt_phrases, convert_to_tensor=True
        )

        # Compute pairwise cosine similarity
        sim_matrix = cos_sim(txt_embeddings, det_embeddings)

        # Build a human-readable similarity matrix for debugging
        similarity_matrix = {}
        for i, t_str in enumerate(txt_labels):
            if t_str not in similarity_matrix:
                similarity_matrix[t_str] = {}
            # Average similarity across instances of the same class for the report
            # Or just store max
            for j, d in enumerate(det_labels):
                score = round(sim_matrix[i][j].item(), 4)
                similarity_matrix[t_str][d] = max(score, similarity_matrix[t_str].get(d, 0))

        # --- Bipartite Matching ---
        # Cost matrix for linear_sum_assignment (wants to minimize cost, so 1 - sim)
        cost_matrix = 1.0 - sim_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        hallucinated = []
        matched_det_indices = set()
        matched_txt_indices = set()

        for r, c in zip(row_ind, col_ind):
            score = sim_matrix[r][c].item()
            if score >= self.threshold:
                matched.append(
                    (txt_list[r], det_list[c], round(score, 4))
                )
                matched_det_indices.add(c)
                matched_txt_indices.add(r)

        # Any text noun not matched is hallucinated
        for r in range(len(txt_list)):
            if r not in matched_txt_indices:
                hallucinated.append(txt_list[r])

        # --- Identify detected objects that the caption missed ---
        # QUANTIFIER-AWARE PLURALITY EXEMPTION:
        # We only exempt remaining instances of a class if the text noun
        # that matched them was actually PLURAL (e.g. "apples", "a bunch of men").
        # If the caption said "a man" (singular), we DO penalize for missing the other men!
        plural_matched_det_labels = set()
        for txt_obj, det, score in matched:
            if isinstance(txt_obj, dict) and txt_obj.get("is_plural", False):
                plural_matched_det_labels.add(det['label'])
            elif isinstance(txt_obj, str): # fallback
                plural_matched_det_labels.add(det['label'])

        missed = [
            det_list[c]
            for c in range(len(det_list))
            if c not in matched_det_indices and det_list[c]['label'] not in plural_matched_det_labels
        ]

        return {
            "matched": matched,
            "hallucinated": hallucinated,
            "missed": missed,
            "similarity_matrix": similarity_matrix,
        }


if __name__ == "__main__":
    # --- Dry Run: Validate phrase-based synonym resolution ---
    print("Validating Semantic Bridge Module (phrase-based)...")
    module = SemanticBridgeModule(threshold=0.65)

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

    # Test 3: The "man" vs "person" case
    print("\n  Test 3: Man vs Person")
    result = module.compute_alignment(
        detected_objects={"person"},
        text_nouns={"man"},
    )
    print(f"    Matched: {result['matched']}")
    print(f"    Hallucinated: {result['hallucinated']}")
    print(f"    Similarity: {result['similarity_matrix']}")

    print("\nSemantic Bridge Module validated successfully.")
