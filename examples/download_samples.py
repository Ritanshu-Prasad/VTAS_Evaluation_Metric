"""
Downloads 100 sample images from the COCO validation dataset.
Stores them locally in assets/samples so you can visually browse them
and test the VTAS metric.
"""

from datasets import load_dataset
import os
import json

def main():
    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "samples"))
    os.makedirs(sample_dir, exist_ok=True)
    
    print(f"Downloading COCO validation subset (100 images) to {sample_dir}...")
    
    # Load dataset (streams data so we don't download the massive full dataset)
    dataset = load_dataset("HuggingFaceM4/COCO", split="validation", streaming=True, trust_remote_code=True)
    
    dataset_iter = iter(dataset)
    
    ground_truths = {}
    
    for i in range(100):
        item = next(dataset_iter)
        img = item["image"]
        
        # Ensure standard RGB format
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img_filename = f"coco_val_{i:03d}.jpg"
        img_path = os.path.join(sample_dir, img_filename)
        img.save(img_path)
        
        # Save human reference captions for later analysis
        raw_caps = item["sentences"]["raw"]
        ground_truths[img_filename] = raw_caps if isinstance(raw_caps, list) else [raw_caps]
        
        if (i + 1) % 20 == 0:
            print(f"Downloaded {i + 1}/100 images...")
            
    # Save ground truths to a JSON file
    gt_path = os.path.join(sample_dir, "ground_truths.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truths, f, indent=4)
        
    print(f"\nSuccessfully saved 100 images and references to {sample_dir}")

if __name__ == "__main__":
    main()
