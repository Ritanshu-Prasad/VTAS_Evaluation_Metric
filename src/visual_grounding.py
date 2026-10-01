"""
Visual Grounding Module — The "Eyes" of VTAS.

Leverages a pre-trained DETR (DEtection TRansformer) model to extract
a set of object labels physically present in a given image. This module
serves as the automated ground truth for the VTAS scoring engine,
eliminating the need for expensive human-annotated object tags.

Architecture:
    facebook/detr-resnet-50 → 41M parameters, COCO-trained (91 categories).
"""

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch


class VisualGroundingModule:
    """
    Extracts a set of detected object labels from a raw image using
    a pre-trained DETR object detection model.

    Attributes:
        processor: The DETR image processor for input normalization.
        model: The pre-trained DETR model for object detection.
        confidence_threshold: Minimum confidence score to accept a detection.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initializes the Visual Grounding Module.

        Args:
            confidence_threshold: Detections below this confidence are
                discarded to reduce noise. Default 0.7 balances precision
                and recall for COCO-style images.
        """
        self.confidence_threshold = confidence_threshold

        model_id = "facebook/detr-resnet-50"
        self.processor = DetrImageProcessor.from_pretrained(model_id)
        self.model = DetrForObjectDetection.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image: Image.Image) -> set:
        """
        Performs object detection on the input image and returns a set
        of unique object labels that exceed the confidence threshold.

        Args:
            image: A PIL Image in RGB mode.

        Returns:
            A set of lowercase object label strings.
            Example: {'person', 'dog', 'frisbee'}
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # Post-process detections into human-readable labels
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]

        detected_objects = set()
        for score, label_id in zip(results["scores"], results["labels"]):
            label = self.model.config.id2label[label_id.item()].lower()
            detected_objects.add(label)

        return detected_objects


if __name__ == "__main__":
    # --- Dry Run: Validate the module loads and runs correctly ---
    print("Validating Visual Grounding Module...")
    module = VisualGroundingModule()

    # Create a simple test image (solid color — should detect nothing)
    test_image = Image.new("RGB", (640, 480), color="blue")
    result = module.detect(test_image)

    print(f"Detected objects: {result}")
    print("Visual Grounding Module validated successfully.")
