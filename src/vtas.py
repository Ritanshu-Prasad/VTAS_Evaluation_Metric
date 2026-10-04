"""
VTAS (Visual-Truth Alignment Score) — Core Evaluator.

This is the primary interface for the VTAS evaluation framework. It
orchestrates four sub-modules into a two-tier scoring pipeline:

    Tier 1 (Object-Level):  DETR + spaCy + MiniLM (phrase-based)
    Tier 2 (Scene-Level):   CLIP fallback for ungrounded nouns

Usage:
    from vtas import VTASEvaluator

    evaluator = VTASEvaluator()
    result = evaluator.score(
        image_path="path/to/image.jpg",
        caption="A man throws a frisbee to his dog"
    )
    print(result['vtas_score'])

Mathematical Formulation (v1.2 — F-beta Score):
    Given:
        V = set of objects detected by DETR in the image
        T = set of nouns extracted from the generated caption
        tau_1 = MiniLM semantic similarity threshold (default: 0.65)
        tau_2 = CLIP image-text similarity threshold (default: 0.22)

    Tier 1 — Object Grounding (DETR + MiniLM):
        M_1 = {t in T : max_v sim_MiniLM(t, v) >= tau_1}
        U   = T - M_1   (ungrounded nouns after Tier 1)

    Tier 2 — Context Grounding (CLIP):
        M_2 = {u in U : sim_CLIP(image, u) >= tau_2}
        H   = U - M_2   (true hallucinations)

    Object Precision (P):
        P = 1 - (|H| / |T|)   = fraction of caption nouns that are grounded
        When |T| = 0: P = 1.0  (no claims made => no false claims)

    Visual Recall (R):
        R = |{v in V : matched}| / |V|   = fraction of detected objects mentioned
        When |V| = 0: R = 1.0  (nothing to miss => perfect recall)

    VTAS Score (F-beta):
        VTAS = (1 + beta^2) * (P * R) / (beta^2 * P + R)
        When P + R = 0: VTAS = 0.0

        beta = 1.0 (default): Equal weight to precision and recall (F1).
        beta < 1.0: Precision-weighted (penalize hallucinations more).
        beta > 1.0: Recall-weighted (penalize missing objects more).

    The F-beta score is the harmonic mean of P and R, a standard
    combination from Information Retrieval (van Rijsbergen, 1979).
    Unlike a weighted arithmetic mean (alpha*P + beta*R), it has no
    arbitrary weights — the single parameter beta has a precise
    mathematical interpretation as the ratio of importance.

    Range: [0, 1] where 1.0 = perfect alignment, 0.0 = complete failure.
"""

from PIL import Image

from visual_grounding import VisualGroundingModule
from linguistic_extraction import LinguisticExtractionModule
from semantic_bridge import SemanticBridgeModule
from clip_fallback import CLIPFallbackModule


class VTASEvaluator:
    """
    The primary VTAS evaluation class. Combines object detection,
    NLP extraction, semantic bridging, and CLIP-based scene verification
    to produce a single interpretable alignment score.

    Architecture (Two-Tier Grounding):
        Tier 1: DETR detects objects -> MiniLM matches nouns to objects.
        Tier 2: Unmatched nouns are verified against the full image via CLIP.
        Only nouns that fail both tiers are classified as hallucinations.

    Attributes:
        visual_module: The DETR-based object detection module.
        linguistic_module: The spaCy-based noun extraction module.
        semantic_module: The MiniLM-based semantic bridging module.
        clip_module: The CLIP-based scene/context verification module.
    """

    def __init__(
        self,
        detection_confidence: float = 0.7,
        similarity_threshold: float = 0.65,
        clip_threshold: float = 0.22,
        beta: float = 1.0,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initializes the VTAS Evaluator by loading all four sub-modules.

        Args:
            detection_confidence: Minimum DETR confidence to accept an
                object detection. Lower values increase recall but may
                introduce noisy detections.
            similarity_threshold: Minimum cosine similarity for the
                Semantic Bridge (Tier 1, phrase-based) to consider two
                words as equivalent.
            clip_threshold: Minimum CLIP image-text similarity for
                Tier 2 context verification. CLIP scores are inherently
                lower than text-text similarity scores.
            beta: The F-beta parameter controlling the precision-recall
                tradeoff. beta=1.0 (F1) gives equal weight. beta<1.0
                penalizes hallucinations more. beta>1.0 penalizes
                missing objects more. This is the standard van Rijsbergen
                (1979) formulation — no arbitrary weights.
            spacy_model: The spaCy language model for noun extraction.

        Note:
            First-time initialization downloads ~800MB of model weights:
            DETR (~160MB), MiniLM (~80MB), CLIP (~600MB), spaCy (~12MB).
            Subsequent runs use the Hugging Face cache.
        """
        self.beta = beta
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

        print("[VTAS] Initializing CLIP Fallback Module (Tier 2)...")
        self.clip_module = CLIPFallbackModule(
            threshold=clip_threshold
        )

        print("[VTAS] All modules loaded. Evaluator ready.\n")

    def score(self, image_path: str, caption: str) -> dict:
        """
        Evaluates a single image-caption pair using the two-tier
        grounding pipeline and returns a full diagnostic breakdown.

        Pipeline:
            1. DETR extracts objects from the image.
            2. spaCy extracts nouns from the caption (prefix-stripped).
            3. MiniLM bridges vocabulary gaps (Tier 1).
            4. Ungrounded nouns go to CLIP for scene verification (Tier 2).
            5. Final score is computed from the combined results.

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
                - 'matched': List of (noun, object, score) from Tier 1.
                - 'clip_grounded': List of (noun, score) rescued by Tier 2.
                - 'hallucinated': List of nouns that failed both tiers.
                - 'missed': List of detected objects not in the caption.
                - 'similarity_matrix': Full pairwise Tier 1 similarity.
                - 'clip_scores': Full CLIP score dict for Tier 2 nouns.
        """
        # --- Stage 1: Visual Grounding (DETR) ---
        image = Image.open(image_path).convert("RGB")
        detected_objects = self.visual_module.detect(image)

        # --- Stage 2: Linguistic Extraction (spaCy) ---
        text_nouns = self.linguistic_module.extract(caption)

        # --- Stage 3: Semantic Bridging — Tier 1 (MiniLM) ---
        alignment = self.semantic_module.compute_alignment(
            detected_objects=detected_objects,
            text_nouns=text_nouns,
        )

        # --- Stage 4: CLIP Fallback — Tier 2 ---
        # Prevent CLIP Over-counting Loophole: CLIP cannot count. If a caption says "banana" twice, 
        # but DETR only found one, the second "banana" shouldn't be rescued by CLIP just because 
        # a banana exists in the scene.
        matched_text_nouns = {
            txt['text'] if isinstance(txt, dict) else txt 
            for txt, det, score in alignment["matched"]
        }
        
        tier1_hallucinated = alignment["hallucinated"]
        nouns_for_clip = []
        strict_hallucinations = []
        
        for txt_obj in tier1_hallucinated:
            noun = txt_obj['text'] if isinstance(txt_obj, dict) else txt_obj
            if noun in matched_text_nouns:
                # If we already matched this exact noun, any extra copies without DETR boxes are hallucinations
                strict_hallucinations.append(noun)
            else:
                nouns_for_clip.append(noun)

        clip_result = self.clip_module.verify(image, nouns_for_clip)

        # Nouns rescued by CLIP are NOT hallucinations
        clip_grounded = clip_result["grounded"]
        final_hallucinated = [noun for noun, _ in clip_result["hallucinated"]] + strict_hallucinations

        # TIER 2 RECALL RESCUE:
        # If a noun like "woman" was rescued by CLIP, it means it's in the image.
        # It likely corresponds to the huge "person" DETR box that missed Tier 1 threshold (0.62 < 0.65).
        # We find these weak synonyms and remove them from the `missed` penalty list!
        valid_rescued_labels = set()
        for noun, _ in clip_grounded:
            if noun in alignment["similarity_matrix"]:
                # find the best matching DETR label
                best_match = max(alignment["similarity_matrix"][noun].items(), key=lambda x: x[1], default=(None, 0))
                if best_match[1] > 0.30:  # Weak synonym threshold
                    valid_rescued_labels.add(best_match[0])

        final_missed = [
            m for m in alignment["missed"] 
            if m['label'] not in valid_rescued_labels
        ]

        # --- Stage 5: Compute Final VTAS Score (F-beta) ---
        num_detected = len(detected_objects)
        num_text_nouns = len(text_nouns)
        num_hallucinated = len(final_hallucinated)

        # Object Precision: fraction of caption nouns that are grounded.
        # When no nouns were extracted, precision is 1.0 (no false claims).
        precision = (
            1.0 - (num_hallucinated / num_text_nouns)
            if num_text_nouns > 0 else 1.0
        )

        # Saliency-weighted Visual Recall (fixes Scene Penalty):
        # Weight each object by its relative bounding box area.
        total_saliency = sum(obj.get('area', 1.0) for obj in detected_objects)
        missed_saliency = sum(obj.get('area', 1.0) for obj in final_missed)
        
        # TIER 2 RECALL BONUS: Reward the VLM for finding objects that DETR completely missed!
        # If CLIP rescues a noun, it means the VLM saw something DETR missed. We assign it a 
        # virtual bounding box area (e.g. 10%) so it positively contributes to the Recall score.
        clip_bonus = 0.10 * len(clip_grounded)
        total_saliency += clip_bonus
        
        matched_saliency = total_saliency - missed_saliency
        
        recall = (
            matched_saliency / total_saliency
            if total_saliency > 0 else 1.0
        )

        # F-beta Score: harmonic mean of Precision and Recall.
        # beta = 1.0 => F1 (equal weight)
        # beta < 1.0 => precision-weighted (penalize hallucinations)
        # beta > 1.0 => recall-weighted (penalize missing objects)
        beta_sq = self.beta ** 2
        denominator = (beta_sq * precision) + recall
        vtas_score = (
            (1 + beta_sq) * (precision * recall) / denominator
            if denominator > 0 else 0.0
        )

        return {
            "vtas_score": round(vtas_score, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "detected_objects": detected_objects,
            "text_nouns": [t['text'] if isinstance(t, dict) else t for t in text_nouns],
            "matched": [(t['text'] if isinstance(t, dict) else t, d, s) for t, d, s in alignment["matched"]],
            "clip_grounded": clip_grounded,
            "hallucinated": final_hallucinated,
            "missed": final_missed,
            "similarity_matrix": alignment["similarity_matrix"],
            "clip_scores": clip_result["scores"],
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

        scores = [r["vtas_score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        print(f"\n[VTAS] Batch complete. Average VTAS: {avg_score:.4f}")
        return results


def _print_report(result: dict) -> None:
    """Formats and prints a human-readable VTAS diagnostic report."""
    print("=" * 60)
    print("          VTAS v1.2 DIAGNOSTIC REPORT")
    print("=" * 60)
    print(f"  VTAS Score (F-beta): {result['vtas_score']:.4f}")
    print(f"  Object Precision:    {result['precision']:.4f}")
    print(f"  Visual Recall:       {result['recall']:.4f}")
    print("-" * 60)
    det_labels = [d['label'] if isinstance(d, dict) else d for d in result['detected_objects']]
    print(f"  DETR Detected:       {det_labels}")
    print(f"  Caption Nouns:       {result['text_nouns']}")
    print("-" * 60)
    
    matched_strs = [f"{t}~{d['label'] if isinstance(d, dict) else d} ({s:.2f})" for t, d, s in result['matched']]
    print(f"  Tier 1 Matched:      {matched_strs}")
    print(f"  Tier 2 CLIP Rescued: {result['clip_grounded']}")
    print(f"  Hallucinated:        {result['hallucinated']}")
    
    missed_labels = [m['label'] if isinstance(m, dict) else m for m in result['missed']]
    print(f"  Missed:              {missed_labels}")
    print("-" * 60)
    print("  Tier 1 Similarity Matrix:")
    for noun, scores in result["similarity_matrix"].items():
        print(f"    {noun}: {scores}")
    print("-" * 60)
    print("  Tier 2 CLIP Scores:")
    for noun, score in result["clip_scores"].items():
        print(f"    {noun}: {score}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  VTAS v1.1 — Interactive Evaluation")
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
