"""
Batch testing script for VTAS.

Runs the Base BLIP model (which is known to hallucinate occasionally)
on the 100 local sample images. It scores them using VTAS, sorts them,
and automatically generates visual infographics for the Best and Worst
performers so you can embed them in your README.
"""

import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from vtas import VTASEvaluator
from generate_illustrations import generate_illustration

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Booting up Batch Testing on {device} ---")
    
    # 1. Load Base BLIP Model (Great for testing because it hallucinates more than fine-tuned models)
    print("Loading Base BLIP Model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    model.eval()
    
    # 2. Load VTAS Evaluator
    evaluator = VTASEvaluator()
    
    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "samples"))
    
    if not os.path.exists(sample_dir):
        print(f"Error: {sample_dir} not found. Run download_samples.py first!")
        return

    # Collect images
    image_files = [f for f in os.listdir(sample_dir) if f.endswith(".jpg")]
    image_files.sort()
    
    results = []
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for img_file in image_files:
        img_path = os.path.join(sample_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        
        # Generate Caption
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        
        # Run VTAS
        vtas_result = evaluator.score(img_path, caption)
        
        # Store data
        results.append({
            "image_file": img_file,
            "image_path": img_path,
            "caption": caption,
            "vtas_score": vtas_result["vtas_score"],
            "visual_recall": vtas_result["visual_recall"],
            "hallucination_rate": vtas_result["hallucination_rate"]
        })
        print(f"[{img_file}] Score: {vtas_result['vtas_score']:.2f} | Caption: {caption}")
        
    # --- Analysis & Illustration Generation ---
    print("\nSorting results and generating infographics for extreme cases...")
    
    # Sort by VTAS score
    results.sort(key=lambda x: x["vtas_score"])
    
    worst_cases = results[:3]   # Bottom 3
    best_cases = results[-3:]   # Top 3 (reverse to get absolute best first)
    best_cases.reverse()
    
    # Generate illustrations for Worst 3
    for i, res in enumerate(worst_cases):
        out_name = f"worst_case_{i+1}.png"
        generate_illustration(
            evaluator, 
            res["image_path"], 
            res["caption"], 
            f"VTAS Example: Hallucination/Low Score", 
            out_name
        )
        
    # Generate illustrations for Best 3
    for i, res in enumerate(best_cases):
        out_name = f"best_case_{i+1}.png"
        generate_illustration(
            evaluator, 
            res["image_path"], 
            res["caption"], 
            f"VTAS Example: Perfect Alignment", 
            out_name
        )
        
    print("\nBatch processing complete! Infographics saved to the assets/ folder.")

if __name__ == "__main__":
    main()
