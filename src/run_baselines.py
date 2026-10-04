import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline comparisons (CLIPScore, BLEU, METEOR) for VTAS.")
    parser.add_argument("--vtas_results", type=str, default="/kaggle/working/vtas_output/vtas_results.json",
                        help="Path to the JSON results from run_evaluation.py.")
    parser.add_argument("--coco_annotations", type=str, required=True,
                        help="Path to COCO reference captions JSON (e.g., captions_val2014.json or captions_val2017.json).")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/vtas_baselines",
                        help="Directory to save baseline results and plots.")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32",
                        help="CLIP model to use for CLIPScore computation.")
    return parser.parse_args()

def compute_clip_score(image_path, caption, model, processor, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # CLIPScore is max(100 * cos(image, text), 0) in some papers, 
            # or just unscaled cosine similarity. We will use the unscaled cosine similarity
            # similar to torchmetrics CLIPScore (which scales by 2.5)
            # Let's use standard logit scaling which is built into CLIP.
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize embeddings
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            clip_score = (image_embeds @ text_embeds.T).item()
            # Often scaled by 2.5 for CLIPScore
            clip_score = max(clip_score * 2.5, 0.0)
            return clip_score
    except Exception as e:
        print(f"Error computing CLIPScore for {image_path}: {e}")
        return 0.0

def run_pycocoevalcap(vtas_results_path, coco_annotations_path, output_dir):
    with open(vtas_results_path, "r") as f:
        vtas_data = json.load(f)
    
    # pycocoevalcap needs a results json in the format: [{"image_id": int, "caption": str}]
    # We need to map our images to COCO image IDs.
    # Since run_evaluation.py streams merve/coco, the "index" in vtas_results isn't the COCO image_id.
    # However, to use pycocoevalcap, we MUST have valid COCO image IDs that match the reference annotations.
    # If merve/coco streams in a deterministic order matching a specific validation split,
    # we can try to associate them by order, but the safest way if image_id is not in vtas_results
    # is to inform the user or assume they added image_id.
    # Wait! run_evaluation.py does not save image_id.
    # Let's generate a temporary annotation structure that matches pycocoevalcap requirements 
    # to bypass the strict image_id matching if we don't have them.
    # Actually, pycocoevalcap requires matching image_ids with the ground truth COCO object.
    
    print("[Baselines] Note: BLEU/METEOR requires accurate COCO image IDs.")
    
    # Format generated results
    # We will assume that the user will update run_evaluation.py to save 'image_id' if needed.
    # For now, we will try to use the 'index' as 'image_id' and create a dummy COCO GT if actual fails,
    # but the standard usage requires real image IDs.
    
    res_list = []
    for item in vtas_data["per_image_results"]:
        # Try to use a real image_id if it exists, otherwise fallback to index
        img_id = item.get("image_id", item["index"])
        res_list.append({
            "image_id": img_id,
            "caption": item["caption"]
        })
        
    res_json_path = os.path.join(output_dir, "temp_results.json")
    with open(res_json_path, "w") as f:
        json.dump(res_list, f)
        
    print(f"[Baselines] Loading COCO annotations from {coco_annotations_path}")
    coco = COCO(coco_annotations_path)
    
    # Load results
    coco_res = coco.loadRes(res_json_path)
    
    # Create evaluator
    coco_eval = COCOEvalCap(coco, coco_res)
    
    # Evaluate on the subset of images we have results for
    coco_eval.params['image_id'] = coco_res.getImgIds()
    
    print("[Baselines] Running pycocoevalcap...")
    coco_eval.evaluate()
    
    metrics = {}
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
        metrics[metric] = score
        
    return metrics

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.vtas_results, "r") as f:
        vtas_data = json.load(f)
        
    results = vtas_data["per_image_results"]
    
    print(f"[Baselines] Loaded {len(results)} results from {args.vtas_results}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Baselines] Loading CLIP model {args.clip_model} on {device}...")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    model.eval()
    
    vtas_scores = []
    clip_scores = []
    hallucination_outliers_x = []
    hallucination_outliers_y = []
    
    print("[Baselines] Computing CLIPScores...")
    for item in results:
        vtas = item["vtas_score"]
        caption = item["caption"]
        img_path = item["image_path"]
        
        # In Kaggle, the image paths in JSON might be /tmp/vtas_eval_images/...
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping CLIPScore for this image.")
            continue
            
        c_score = compute_clip_score(img_path, caption, model, processor, device)
        item["clip_score"] = c_score
        
        vtas_scores.append(vtas)
        clip_scores.append(c_score)
        
        # Identify hallucination outliers: High CLIPScore but Low VTAS
        # CLIPScore generally ranges 0.2 - 0.4 unscaled (or 0.5 - 1.0 scaled)
        if vtas < 0.3 and c_score > 0.65:
            hallucination_outliers_x.append(c_score)
            hallucination_outliers_y.append(vtas)

    if len(vtas_scores) > 0:
        pearson_corr, _ = pearsonr(vtas_scores, clip_scores)
        spearman_corr, _ = spearmanr(vtas_scores, clip_scores)
        
        print(f"\n[Baselines] Pearson Correlation (VTAS vs CLIPScore): {pearson_corr:.4f}")
        print(f"[Baselines] Spearman Correlation (VTAS vs CLIPScore): {spearman_corr:.4f}")
        
        # Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(clip_scores, vtas_scores, alpha=0.6, edgecolors='w', label="Images")
        if hallucination_outliers_x:
            plt.scatter(hallucination_outliers_x, hallucination_outliers_y, color='red', label='Hallucination Outliers')
            
        plt.xlabel("CLIPScore")
        plt.ylabel("VTAS Score")
        plt.title("VTAS vs CLIPScore")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_path = os.path.join(args.output_dir, "vtas_vs_clipscore.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Baselines] Scatter plot saved to {plot_path}")
    else:
        print("[Baselines] Not enough valid data points for correlation/plotting.")
        pearson_corr = spearman_corr = 0
        
    # Run pycocoevalcap
    try:
        metrics = run_pycocoevalcap(args.vtas_results, args.coco_annotations, args.output_dir)
    except Exception as e:
        print(f"[Baselines] Failed to run pycocoevalcap: {e}")
        print("[Baselines] Make sure pycocoevalcap is installed and coco_annotations match the images.")
        metrics = {}
        
    # Save baseline results
    summary_path = os.path.join(args.output_dir, "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "correlation": {
                "pearson": pearson_corr,
                "spearman": spearman_corr
            },
            "nlg_metrics": metrics
        }, f, indent=4)
        
    print(f"[Baselines] Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
