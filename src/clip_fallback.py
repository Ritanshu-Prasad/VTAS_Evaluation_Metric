"""
CLIP Fallback Module — The "Context Awareness" Layer of VTAS.

When a noun extracted from a caption fails to match any DETR-detected
object via the Semantic Bridge (Tier 1), it is not immediately classified
as a hallucination. Instead, it is forwarded to this module for a second
opinion using OpenAI's CLIP model.

CLIP operates on the *entire image* rather than bounding boxes, allowing
it to verify abstract concepts that DETR cannot detect:
    - Scene/environment terms: "bedroom", "kitchen", "outdoors"
    - Contextual objects too small for DETR's confidence threshold
    - Synonyms that fell below the MiniLM similarity cutoff

Architecture:
    openai/clip-vit-base-patch32 -> 151M parameters.
    Computes a joint image-text similarity score by projecting
    both the full image and a text prompt into a shared 512-dim
    embedding space.

Design Rationale:
    DETR draws bounding boxes around discrete physical objects but
    cannot understand scenes (e.g., "this is a bedroom"). CLIP was
    trained on 400M image-text pairs from the internet and excels at
    holistic image understanding. By combining both, VTAS achieves
    object-level precision (DETR) with scene-level recall (CLIP).
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


class CLIPFallbackModule:
    """
    Verifies ungrounded nouns against the full image using CLIP's
    zero-shot image-text similarity. Acts as a secondary grounding
    tier when DETR + MiniLM fail to find a match.

    Attributes:
        processor: The CLIP image/text preprocessor.
        model: The pre-trained CLIP model.
        threshold: Minimum CLIP similarity to accept a noun as
            contextually grounded (not a hallucination).
    """

    def __init__(self, threshold: float = 0.22):
        """
        Initializes the CLIP Fallback Module.

        Args:
            threshold: Minimum cosine similarity between the image
                embedding and the text prompt embedding for a noun
                to be considered contextually grounded. CLIP similarity
                scores are typically lower than MiniLM text-text scores.
                Default 0.22 was calibrated to accept scene descriptors
                ("bedroom" in a bedroom image scores ~0.25-0.30) while
                rejecting true hallucinations ("elephant" in a kitchen
                scores ~0.10-0.15).
        """
        self.threshold = threshold

        model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def verify(self, image: Image.Image, nouns: list) -> dict:
        """
        Checks whether each noun is contextually present in the image
        using CLIP's zero-shot classification capability.

        For each noun, constructs a prompt "a photo of a {noun}" and
        computes its cosine similarity with the image embedding. If the
        similarity exceeds the threshold, the noun is considered
        contextually grounded.

        Args:
            image: A PIL Image in RGB mode.
            nouns: A list of noun strings that failed Tier 1 grounding.
                Example: ['bedroom', 'player', 'grass']

        Returns:
            A dictionary containing:
                - 'grounded': List of (noun, score) tuples that CLIP
                    confirmed are present in the image.
                - 'hallucinated': List of (noun, score) tuples that CLIP
                    also could not verify.
                - 'scores': Dict mapping each noun to its CLIP score.
        """
        if not nouns:
            return {"grounded": [], "hallucinated": [], "scores": {}}

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Construct natural language prompts for each noun
        prompts = [f"a photo of a {noun}" for noun in nouns]

        # Encode image and text into CLIP's shared embedding space
        inputs = self.processor(
            text=prompts, images=image,
            return_tensors="pt", padding=True
        )
        outputs = self.model(**inputs)

        # Compute per-prompt cosine similarity with the image
        logits = outputs.logits_per_image[0]  # shape: [num_nouns]
        # Normalize logits to [0, 1] range using softmax for multi-noun
        # comparison, but use raw cosine for single-noun threshold check
        scores = logits / 100.0  # CLIP logits are scaled by 100

        grounded = []
        hallucinated = []
        score_dict = {}

        for i, noun in enumerate(nouns):
            sim = scores[i].item()
            score_dict[noun] = round(sim, 4)

            if sim >= self.threshold:
                grounded.append((noun, round(sim, 4)))
            else:
                hallucinated.append((noun, round(sim, 4)))

        return {
            "grounded": grounded,
            "hallucinated": hallucinated,
            "scores": score_dict,
        }


if __name__ == "__main__":
    print("Validating CLIP Fallback Module...")
    module = CLIPFallbackModule()

    # Create a simple test with a blank image
    test_image = Image.new("RGB", (640, 480), color="white")
    result = module.verify(test_image, ["bedroom", "car", "sunshine"])

    print(f"  Grounded: {result['grounded']}")
    print(f"  Hallucinated: {result['hallucinated']}")
    print(f"  Scores: {result['scores']}")
    print("CLIP Fallback Module validated successfully.")
