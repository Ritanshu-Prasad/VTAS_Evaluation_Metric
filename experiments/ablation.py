
import json
import math

with open('../Preprint/Phase 1/results/vtas_output/vtas_results.json') as f:
    data = json.load(f)

golden_scores = []
no_sal_scores = []
no_tier2_scores = []

for r in data['per_image_results']:
    golden_scores.append(r['vtas_score'])
    
    precision = r['precision']
    recall = r['recall']
    
    matched = r['matched']
    clip_grounded = r['clip_grounded']
    hallucinated = r['hallucinated']
    missed = r['missed']
    
    num_matched = len(matched)
    num_clip = len(clip_grounded)
    num_hallucinated = len(hallucinated)
    num_missed = len(missed)
    
    # --- Reverse Engineer text_nouns ---
    if precision < 1.0 and num_hallucinated > 0:
        num_text_nouns = round(num_hallucinated / (1.0 - precision))
    else:
        num_text_nouns = num_matched + num_clip
        
    # --- Reverse Engineer total_saliency ---
    matched_sal_no_tier2 = sum(m[1].get('area', 1.0) for m in matched)
    missed_sal = sum(m.get('area', 1.0) for m in missed)
    
    # recall = (matched_sal_no_tier2 + 0.10 * total_sal * num_clip) / total_sal
    # total_sal = matched_sal_no_tier2 / (recall - 0.10 * num_clip)
    denom = recall - 0.10 * num_clip
    if denom > 0:
        total_sal = matched_sal_no_tier2 / denom
    else:
        total_sal = matched_sal_no_tier2 + missed_sal # fallback
        
    if total_sal == 0:
        total_sal = 1.0
        
    # ==========================
    # Ablation 1: No Saliency
    # Treat area as 1.0 for all boxes
    # ==========================
    total_sal_nosal = num_matched + num_missed
    if total_sal_nosal == 0: total_sal_nosal = 1.0
    matched_sal_nosal = num_matched + (0.10 * total_sal_nosal * num_clip)
    
    rec_nosal = min(1.0, matched_sal_nosal / total_sal_nosal)
    
    # vtas formula (beta=1)
    denom_nosal = precision + rec_nosal
    score_nosal = 2 * (precision * rec_nosal) / denom_nosal if denom_nosal > 0 else 0.0
    no_sal_scores.append(score_nosal)
    
    # ==========================
    # Ablation 2: No Tier 2
    # Move clip_grounded to hallucinated, remove bonus
    # ==========================
    new_num_hall = num_hallucinated + num_clip
    prec_notier2 = max(0.0, 1.0 - (new_num_hall / num_text_nouns)) if num_text_nouns > 0 else 1.0
    
    rec_notier2 = min(1.0, matched_sal_no_tier2 / total_sal)
    
    denom_notier2 = prec_notier2 + rec_notier2
    score_notier2 = 2 * (prec_notier2 * rec_notier2) / denom_notier2 if denom_notier2 > 0 else 0.0
    no_tier2_scores.append(score_notier2)
    
print(f'Golden VTAS: {sum(golden_scores)/len(golden_scores):.4f}')
print(f'VTAS w/o Saliency: {sum(no_sal_scores)/len(no_sal_scores):.4f}')
print(f'VTAS w/o Tier 2: {sum(no_tier2_scores)/len(no_tier2_scores):.4f}')

