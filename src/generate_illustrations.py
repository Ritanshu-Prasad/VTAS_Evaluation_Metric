"""
VTAS Illustration Generator.

Generates infographic-style images that break down how VTAS scores
an image-caption pair across both grounding tiers. Designed to be
saved to disk for README documentation and offline review.

Output includes:
    - The original image
    - DETR detections and spaCy-extracted nouns
    - Tier 1 matches (MiniLM) and Tier 2 rescues (CLIP)
    - Final hallucination list and VTAS score breakdown
"""

import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend for headless servers
import matplotlib.pyplot as plt
from PIL import Image
import os
import textwrap

from vtas import VTASEvaluator


def generate_illustration(
    evaluator: VTASEvaluator,
    image_path: str,
    caption: str,
    title: str,
    output_path: str
):
    """
    Runs the evaluator and generates a matplotlib figure combining
    the image with the VTAS diagnostic breakdown.

    Args:
        evaluator: An initialized VTASEvaluator instance.
        image_path: Path to the image file.
        caption: The generated caption to evaluate.
        title: Title text for the infographic.
        output_path: Full path (including filename) where the PNG
            will be saved. Supports both absolute and relative paths.
    """
    print(f"Generating illustration for: {title}...")

    # 1. Run the metric
    result = evaluator.score(image_path, caption)

    # 2. Setup the Matplotlib figure
    fig = plt.figure(figsize=(16, 8))
    fig.patch.set_facecolor('#f8f9fa')

    # Left column: The Image
    ax_img = plt.subplot(1, 2, 1)
    img = Image.open(image_path)
    ax_img.imshow(img)
    ax_img.axis('off')
    ax_img.set_title(title, fontsize=16, fontweight='bold', pad=15)

    # Right column: The Text Breakdown
    ax_text = plt.subplot(1, 2, 2)
    ax_text.axis('off')

    y_pos = 0.98
    spacing = 0.065

    # Header: Caption
    ax_text.text(0.0, y_pos, "Generated Caption:", fontsize=11, color='gray')
    y_pos -= spacing * 0.7
    wrapped = textwrap.fill(f'"{caption}"', width=50)
    ax_text.text(0.0, y_pos, wrapped, fontsize=12, fontweight='bold',
                 style='italic')
    y_pos -= spacing * 1.3

    # DETR Detections
    ax_text.text(0.0, y_pos, "1. DETR Detected Objects:",
                 fontsize=11, fontweight='bold')
    y_pos -= spacing * 0.7
    det_str = textwrap.fill(str(result['detected_objects']), width=55)
    ax_text.text(0.03, y_pos, det_str, fontsize=10, color='#1f77b4')
    y_pos -= spacing

    # NLP Extracted Nouns
    ax_text.text(0.0, y_pos, "2. Extracted Nouns (after filtering):",
                 fontsize=11, fontweight='bold')
    y_pos -= spacing * 0.7
    ax_text.text(0.03, y_pos, str(result['text_nouns']),
                 fontsize=10, color='#ff7f0e')
    y_pos -= spacing

    # Tier 1: MiniLM Matches
    ax_text.text(0.0, y_pos, "3. Tier 1 - MiniLM Matches:",
                 fontsize=11, fontweight='bold')
    y_pos -= spacing * 0.7
    matched_str = ", ".join(
        [f"{t}~{d} ({s:.2f})" for t, d, s in result['matched']]
    )
    ax_text.text(0.03, y_pos,
                 f"Matched: {matched_str if matched_str else 'None'}",
                 fontsize=9, color='green')
    y_pos -= spacing

    # Tier 2: CLIP Rescues
    ax_text.text(0.0, y_pos, "4. Tier 2 - CLIP Scene Verification:",
                 fontsize=11, fontweight='bold')
    y_pos -= spacing * 0.7
    clip_str = ", ".join(
        [f"{n} ({s:.2f})" for n, s in result['clip_grounded']]
    )
    ax_text.text(0.03, y_pos,
                 f"Rescued: {clip_str if clip_str else 'None'}",
                 fontsize=9, color='#2ca02c')
    y_pos -= spacing * 0.7

    # Hallucinations
    halluc_str = ", ".join(result['hallucinated'])
    ax_text.text(0.03, y_pos,
                 f"Hallucinations: {halluc_str if halluc_str else 'None'}",
                 fontsize=9, color='red')
    y_pos -= spacing * 0.7

    # Missed
    missed_str = ", ".join(result['missed'])
    ax_text.text(0.03, y_pos,
                 f"Missed by caption: {missed_str if missed_str else 'None'}",
                 fontsize=9, color='orange')
    y_pos -= spacing * 1.3

    # Final Score Box
    box_props = dict(
        boxstyle='round,pad=0.5', facecolor='#e6f2ff', edgecolor='#b3d9ff'
    )
    score_text = (
        f"Object Precision: {result['precision']:.2f}\n"
        f"Visual Recall: {result['recall']:.2f}\n"
        f"VTAS Score (F1): {result['vtas_score']:.2f}"
    )
    ax_text.text(0.0, y_pos, score_text, fontsize=13,
                 fontweight='bold', bbox=box_props)

    # Save — ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import sys

    print("Loading VTAS Evaluator (this takes a moment)...")
    evaluator = VTASEvaluator()

    if len(sys.argv) == 4:
        image_path = sys.argv[1]
        caption = sys.argv[2]
        out_path = sys.argv[3]
        generate_illustration(
            evaluator, image_path, caption, "VTAS Evaluation", out_path
        )
    else:
        print(
            "Usage: python generate_illustrations.py "
            "<image_path> \"<caption>\" <output_path.png>"
        )
