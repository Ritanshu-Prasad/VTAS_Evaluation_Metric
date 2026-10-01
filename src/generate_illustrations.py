"""
VTAS Illustration Generator.

This script automatically generates beautiful infographic-style images 
that break down exactly how VTAS scores an image-caption pair. 
These graphics are designed to be embedded in the README.md to visually 
explain the concepts of Visual Recall, Hallucination, and Semantic Bridging.
"""

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
    output_filename: str
):
    """
    Runs the evaluator and generates a matplotlib figure combining 
    the image with the VTAS diagnostic breakdown.
    """
    print(f"Generating illustration for: {title}...")
    
    # 1. Run the metric
    result = evaluator.score(image_path, caption)
    
    # 2. Setup the Matplotlib figure
    fig = plt.figure(figsize=(12, 6))
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
    
    y_pos = 0.95
    spacing = 0.08
    
    # Header
    ax_text.text(0.0, y_pos, "Model Generated Caption:", fontsize=12, color='gray')
    y_pos -= spacing
    
    # Wrap caption text so it fits
    wrapped_caption = textwrap.fill(f'"{caption}"', width=45)
    ax_text.text(0.0, y_pos, wrapped_caption, fontsize=14, fontweight='bold', style='italic')
    y_pos -= (spacing * 1.5)
    
    # DETR Output
    ax_text.text(0.0, y_pos, "1. DETR (Visual Ground Truth):", fontsize=12, fontweight='bold')
    y_pos -= (spacing * 0.8)
    ax_text.text(0.05, y_pos, str(result['detected_objects']), fontsize=12, color='#1f77b4')
    y_pos -= spacing
    
    # NLP Output
    ax_text.text(0.0, y_pos, "2. NLP (Extracted Nouns):", fontsize=12, fontweight='bold')
    y_pos -= (spacing * 0.8)
    ax_text.text(0.05, y_pos, str(result['text_nouns']), fontsize=12, color='#ff7f0e')
    y_pos -= spacing
    
    # Bridge Output
    ax_text.text(0.0, y_pos, "3. Semantic Bridge:", fontsize=12, fontweight='bold')
    y_pos -= (spacing * 0.8)
    
    matched_str = ", ".join([f"{t}↔{d}" for t, d, s in result['matched']])
    ax_text.text(0.05, y_pos, f"✅ Matches: {matched_str if matched_str else 'None'}", fontsize=11, color='green')
    y_pos -= (spacing * 0.8)
    
    halluc_str = ", ".join(result['hallucinated'])
    ax_text.text(0.05, y_pos, f"❌ Hallucinations: {halluc_str if halluc_str else 'None'}", fontsize=11, color='red')
    y_pos -= (spacing * 1.5)
    
    # Final Score Box
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e6f2ff', edgecolor='#b3d9ff')
    score_text = (
        f"Visual Recall: {result['visual_recall']:.2f}\n"
        f"Hallucination Rate: {result['hallucination_rate']:.2f}\n"
        f"Final VTAS Score: {result['vtas_score']:.2f}"
    )
    ax_text.text(0.0, y_pos, score_text, fontsize=14, fontweight='bold', bbox=box_props)
    
    # Save the figure
    plt.tight_layout()
    output_path = os.path.join("..", "assets", output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    import sys
    
    print("Loading VTAS Evaluator (this takes a moment)...")
    evaluator = VTASEvaluator()
    
    # You can pass arguments from the CLI or add hardcoded examples here
    if len(sys.argv) == 4:
        image_path = sys.argv[1]
        caption = sys.argv[2]
        out_name = sys.argv[3]
        generate_illustration(evaluator, image_path, caption, "VTAS Evaluation", out_name)
    else:
        print("Usage: python generate_illustrations.py <image_path> \"<caption>\" <output_filename.png>")
