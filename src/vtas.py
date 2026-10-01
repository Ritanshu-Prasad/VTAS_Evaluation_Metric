"""
VTAS (Visual-Truth Alignment Score) — Core Evaluator.

This is the primary interface for the VTAS evaluation framework. It
orchestrates the three sub-modules (Visual Grounding, Linguistic
Extraction, Semantic Bridge) into a single scoring pipeline.

Usage:
    from vtas import VTASEvaluator

    evaluator = VTASEvaluator()
    result = evaluator.score(
        image_path="path/to/image.jpg",
        caption="A man throws a frisbee to his dog"
    )
    print(result['vtas_score'])

Mathematical Formulation:
    Given:
        V = set of objects detected by DETR in the image
        T = set of nouns extracted from the generated caption
        tau = semantic similarity threshold (default: 0.75)

    Visual Recall (VR):
        VR = |{v in V : max_t sim(t,v) >= tau}| / |V|

    Hallucination Rate (HR):
        HR = |{t in T : max_v sim(t,v) < tau}| / |T|

    VTAS Score:
        VTAS = VR * (1 - HR)

    Range: [0, 1] where 1.0 = perfect alignment, 0.0 = complete failure.
"""

from PIL import Image

from visual_grounding import VisualGroundingModule
from linguistic_extraction import LinguisticExtractionModule
from semantic_bridge import SemanticBridgeModule


class VTASEvaluator:
    """
    The primary VTAS evaluation class. Combines object detection,
    NLP extraction, and semantic bridging to produce a single
    interpretable alignment score for an image-caption pair.

    Attributes:
        visual_module: The DETR-based object detection module.
        linguistic_module: The spaCy-based noun extraction module.
        semantic_module: The MiniLM-based semantic bridging module.
    """

    def __init__(
        self,
        detection_confidence: float = 0.7,
        similarity_threshold: float = 0.75,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initializes the VTAS Evaluator by loading all three sub-modules.

        Args:
            detection_confidence: Minimum DETR confidence to accept an
                object detection. Lower values increase recall but may
                introduce noisy detections.
            similarity_threshold: Minimum cosine similarity for the
                Semantic Bridge to consider two words as equivalent.
            spacy_model: The spaCy language model for noun extraction.

        Note:
            First-time initialization will download ~500MB of model
            weights (DETR: ~160MB, MiniLM: ~80MB, spaCy: ~12MB).
            Subsequent runs use the cached models.
        """
        print("[VTAS] Initializing Visual Grounding Module (DETR)...")
        self.visual_module = VisualGroundingModule(
            confidence_threshold=detection_confidence
        )

        print("[VTAS] Initializing Linguistic Extraction Module (spaCy)...")
        self.linguistic_module = LinguisticExtractionModule(
            model_name=spacy_model
        )

        print("[VTAS] Initializing Semantic Bridge Module (MiniLM)...")
        self.semantic_module = SemanticBridgeModule(
            threshold=similarity_threshold
        )

        print("[VTAS] All modules loaded. Evaluator ready.\n")

    def score(self, image_path: str, caption: str) -> dict:
        """
        Evaluates a single image-caption pair and returns the full
        VTAS diagnostic breakdown.

        Args:
            image_path: Absolute or relative path to the image file.
            caption: The generated caption string to evaluate.

        Returns:
            A dictionary containing:
                - 'vtas_score': The final VTAS score (float, 0 to 1).
                - 'visual_recall': Fraction of detected objects mentioned.
                - 'hallucination_rate': Fraction of text nouns not grounded.
                - 'detected_objects': Set of objects found by DETR.
                - 'text_nouns': Set of nouns extracted from the caption.
                - 'matched': List of (noun, object, similarity) tuples.
                - 'hallucinated': List of nouns with no visual grounding.
                - 'missed': List of detected objects not in the caption.
                - 'similarity_matrix': Full pairwise similarity scores.
        """
        # --- Stage 1: Visual Grounding ---
        image = Image.open(image_path).convert("RGB")
        detected_objects = self.visual_module.detect(image)

        # --- Stage 2: Linguistic Extraction ---
        text_nouns = self.linguistic_module.extract(caption)

        # --- Stage 3: Semantic Bridging & Scoring ---
        alignment = self.semantic_module.compute_alignment(
            detected_objects=detected_objects,
            text_nouns=text_nouns,
        )

        # --- Stage 4: Compute Final VTAS Score ---
        num_detected = len(detected_objects)
        num_text_nouns = len(text_nouns)
        num_matched_det = num_detected - len(alignment["missed"])
        num_hallucinated = len(alignment["hallucinated"])

        # Handle edge cases to prevent division by zero
        visual_recall = num_matched_det / num_detected if num_detected > 0 else 0.0
        hallucination_rate = num_hallucinated / num_text_nouns if num_text_nouns > 0 else 0.0

        vtas_score = visual_recall * (1 - hallucination_rate)

        return {
            "vtas_score": round(vtas_score, 4),
            "visual_recall": round(visual_recall, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "detected_objects": detected_objects,
            "text_nouns": text_nouns,
            "matched": alignment["matched"],
            "hallucinated": alignment["hallucinated"],
            "missed": alignment["missed"],
            "similarity_matrix": alignment["similarity_matrix"],
        }

    def score_batch(self, pairs: list) -> list:
        """
        Evaluates a batch of image-caption pairs and returns the
        average VTAS score alongside individual diagnostics.

        Args:
            pairs: A list of dicts, each with keys 'image_path' and
                'caption'. Example:
                [
                    {"image_path": "img1.jpg", "caption": "A dog..."},
                    {"image_path": "img2.jpg", "caption": "A car..."},
                ]

        Returns:
            A list of result dictionaries (one per pair), with
            an 'index' key added to each for traceability.
        """
        results = []
        for idx, pair in enumerate(pairs):
            print(f"[VTAS] Scoring image {idx + 1}/{len(pairs)}...")
            result = self.score(
                image_path=pair["image_path"],
                caption=pair["caption"],
            )
            result["index"] = idx
            results.append(result)

        # Compute aggregate statistics
        scores = [r["vtas_score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        print(f"\n[VTAS] Batch complete. Average VTAS: {avg_score:.4f}")
        return results


def _print_report(result: dict) -> None:
    """Formats and prints a human-readable VTAS diagnostic report."""
    print("=" * 60)
    print("          VTAS DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"  VTAS Score:          {result['vtas_score']:.4f}")
    print(f"  Visual Recall:       {result['visual_recall']:.4f}")
    print(f"  Hallucination Rate:  {result['hallucination_rate']:.4f}")
    print("-" * 60)
    print(f"  DETR Detected:       {result['detected_objects']}")
    print(f"  Caption Nouns:       {result['text_nouns']}")
    print("-" * 60)
    print(f"  ✅ Matched:          {result['matched']}")
    print(f"  ❌ Hallucinated:     {result['hallucinated']}")
    print(f"  ⚠️  Missed:           {result['missed']}")
    print("-" * 60)
    print("  Similarity Matrix:")
    for noun, scores in result["similarity_matrix"].items():
        print(f"    {noun}: {scores}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # --- Interactive Dry Run ---
    print("=" * 60)
    print("  VTAS v1 — Interactive Evaluation")
    print("=" * 60)

    if len(sys.argv) < 3:
        print("\nUsage: python vtas.py <image_path> <caption>")
        print('Example: python vtas.py test.jpg "A man throws a frisbee"')
        sys.exit(1)

    image_path = sys.argv[1]
    caption = " ".join(sys.argv[2:])

    print(f"\n  Image: {image_path}")
    print(f"  Caption: \"{caption}\"\n")

    evaluator = VTASEvaluator()
    result = evaluator.score(image_path=image_path, caption=caption)

    _print_report(result)
