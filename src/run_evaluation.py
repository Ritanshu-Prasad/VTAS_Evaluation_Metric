"""
VTAS Evaluation Runner — Standalone Kaggle Execution Script.

This script is executed directly on Kaggle via:
    !python src/run_evaluation.py --lora_weights /kaggle/input/...

It loads a BLIP model (optionally with LoRA), generates captions for
COCO validation images, scores each with VTAS, and saves all results
and infographics to the output directory for local download.

Following the Decoupled Compute Architecture:
    - GitHub is the Brain (all code lives here).
    - Kaggle is the Muscle (only clones, installs, and runs).
"""

import os
import sys
import json
import shutil
import argparse

import torch
from PIL import Image
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel

# Ensure the src/ directory is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vtas import VTASEvaluator
from generate_illustrations import generate_illustration


def parse_args():
    """Parse command-line arguments for flexible execution."""
    parser = argparse.ArgumentParser(
        description="Run VTAS evaluation on COCO images with a BLIP model."
    )
    parser.add_argument(
        "--lora_weights", type=str, default=None,
        help="Path to LoRA adapter weights directory. If not provided, "
             "the base BLIP model is used (zero-shot)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="/kaggle/working/vtas_output",
        help="Directory to save results JSON and infographics."
    )
    parser.add_argument(
        "--num_images", type=int, default=100,
        help="Number of COCO validation images to evaluate."
    )
    parser.add_argument(
        "--num_best", type=int, default=5,
        help="Number of best-scoring infographics to generate."
    )
    parser.add_argument(
        "--num_worst", type=int, default=5,
        help="Number of worst-scoring infographics to generate."
    )
    return parser.parse_args()


def load_model(lora_weights_path: str = None):
    """
    Loads the BLIP captioning model with optional LoRA adapter.

    Args:
        lora_weights_path: Path to the LoRA weights directory.
            If None or path does not exist, returns the base model.

    Returns:
        Tuple of (model, processor, device).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Runner] Device: {device}")

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    base_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    if lora_weights_path and os.path.exists(lora_weights_path):
        print(f"[Runner] Loading LoRA adapter from {lora_weights_path}...")
        model = PeftModel.from_pretrained(base_model, lora_weights_path)
    else:
        if lora_weights_path:
            print(f"[Runner] WARNING: {lora_weights_path} not found!")
        print("[Runner] Using base BLIP model (zero-shot).")
        model = base_model

    model.to(device)
    model.eval()
    print("[Runner] Model ready.\n")
    return model, processor, device


def generate_caption(model, processor, image, device):
    """
    Generates a caption for a single image and strips any prompt
    engineering artifacts from the output.

    Args:
        model: The BLIP model (with or without LoRA).
        processor: The BLIP processor.
        image: A PIL Image in RGB mode.
        device: The torch device string.

    Returns:
        A cleaned caption string.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True).strip()
    return caption


def run_evaluation(args):
    """
    Main evaluation pipeline. Loads models, iterates over COCO images,
    scores with VTAS, and saves results + infographics to disk.
    """
    # --- Load Models ---
    model, processor, device = load_model(args.lora_weights)

    print("[Runner] Initializing VTAS Evaluator (loading DETR, MiniLM, CLIP)...")
    evaluator = VTASEvaluator(
        detection_confidence=0.7,
        similarity_threshold=0.65,
        clip_threshold=0.22,
    )

    # --- Load Dataset ---
    print("[Runner] Loading COCO validation dataset (streaming)...")
    dataset = load_dataset("merve/coco", split="validation", streaming=True)
    dataset_iter = iter(dataset)

    # --- Prepare Output Directories ---
    infographic_dir = os.path.join(args.output_dir, "infographics")
    os.makedirs(infographic_dir, exist_ok=True)

    temp_img_dir = "/tmp/vtas_eval_images"
    os.makedirs(temp_img_dir, exist_ok=True)

    # --- Evaluate ---
    results = []
    print(f"[Runner] Evaluating {args.num_images} images...\n")

    for i in range(args.num_images):
        item = next(dataset_iter)
        image = item["image"].convert("RGB")

        img_path = os.path.join(temp_img_dir, f"img_{i:03d}.jpg")
        image.save(img_path)

        caption = generate_caption(model, processor, image, device)
        score_dict = evaluator.score(img_path, caption)

        # Save the cleaned caption (prefix stripped by the evaluator)
        # to the JSON for clean output
        clean_caption = evaluator.linguistic_module._strip_prefix(caption)

        results.append({
            "index": i,
            "image_path": img_path,
            "caption": clean_caption,
            "vtas_score": score_dict["vtas_score"],
            "precision": score_dict["precision"],
            "recall": score_dict["recall"],
            "matched": [
                (t, d, float(s)) for t, d, s in score_dict["matched"]
            ],
            "clip_grounded": [
                (n, float(s)) for n, s in score_dict["clip_grounded"]
            ],
            "hallucinated": score_dict["hallucinated"],
            "missed": score_dict["missed"],
        })

        if (i + 1) % 25 == 0:
            avg = sum(r["vtas_score"] for r in results) / len(results)
            print(f"  [{i+1}/{args.num_images}] Running Avg VTAS: {avg:.4f}")

    # --- Compute Summary ---
    avg_vtas = sum(r["vtas_score"] for r in results) / len(results)
    print(f"\n[Runner] Final Average VTAS Score: {avg_vtas:.4f}")

    # --- Save JSON Results ---
    json_path = os.path.join(args.output_dir, "vtas_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "average_vtas": avg_vtas,
            "total_images": args.num_images,
            "per_image_results": results,
        }, f, indent=4)
    print(f"[Runner] Results saved to {json_path}")

    # --- Generate Infographics ---
    results_sorted = sorted(results, key=lambda x: x["vtas_score"])
    worst = results_sorted[:args.num_worst]
    best = results_sorted[-args.num_best:][::-1]

    print(f"\n[Runner] Generating {args.num_worst} worst-case infographics...")
    for i, res in enumerate(worst):
        out_path = os.path.join(
            infographic_dir,
            f"worst_{i+1}_score_{res['vtas_score']:.2f}.png"
        )
        generate_illustration(
            evaluator, res["image_path"], res["caption"],
            f"Low Alignment (VTAS={res['vtas_score']:.2f})", out_path
        )
        shutil.copy(
            res["image_path"],
            os.path.join(infographic_dir, f"worst_{i+1}_raw.jpg")
        )

    print(f"[Runner] Generating {args.num_best} best-case infographics...")
    for i, res in enumerate(best):
        out_path = os.path.join(
            infographic_dir,
            f"best_{i+1}_score_{res['vtas_score']:.2f}.png"
        )
        generate_illustration(
            evaluator, res["image_path"], res["caption"],
            f"High Alignment (VTAS={res['vtas_score']:.2f})", out_path
        )
        shutil.copy(
            res["image_path"],
            os.path.join(infographic_dir, f"best_{i+1}_raw.jpg")
        )

    print(f"\n[Runner] All outputs saved to {args.output_dir}/")
    print("[Runner] Download the output folder from the Kaggle sidebar.")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
