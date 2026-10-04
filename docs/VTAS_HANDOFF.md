# VTAS Project — Complete Handoff Document

> **Purpose:** This document contains everything needed for a new chat session to continue work on the VTAS arXiv preprint and future metric refinement. Read this entire document before taking any action.

---

## 1. Project Overview

**VTAS (Visual-Truth Alignment Score)** is a reference-free, interpretable evaluation metric for image captioning. It detects object hallucinations by grounding the generated caption against the actual visual content of the image using a two-tier pipeline.

- **GitHub Repo:** https://github.com/Ritanshu-Prasad/VTAS_Evaluation_Metric
- **Local Path:** `c:\Users\prasa\Ritanshu\IITM\Computer Vision\Projects\VTAS_Evaluation_Metric\vtas-metric\`
- **Current Version:** v1.2 (Two-Tier Grounding + F-beta Score)
- **Status:** Code finalized, README published, repo is ready to be made public, arXiv account created.

---

## 2. Architecture (Two-Tier Pipeline)

```
Image ──► DETR (facebook/detr-resnet-50, 41M params)
              │
              ▼
          Detected Objects: {'person', 'motorcycle', 'dog'}
                                                              ┐
Caption ──► spaCy (en_core_web_sm)                            │
              │                                                │  Tier 1
              ▼                                                │
          Extracted Nouns: {'man', 'bike', 'puppy'}            │
              │                                                │
              ▼                                                │
          MiniLM Semantic Bridge (all-MiniLM-L6-v2, 22M)      │
          τ₁ = 0.65 (phrase-based encoding)                   ┘
              │
              ├── Matched: man→person (0.70), bike→motorcycle (0.82)
              └── Unmatched: {'puppy'}
                      │
                      ▼                                       ┐
                  CLIP Fallback (clip-vit-base-patch32, 151M) │ Tier 2
                  τ₂ = 0.22                                   ┘
                      │
                      ├── Rescued (CLIP confirms): puppy (0.28)
                      └── Hallucinated (CLIP rejects): —
                              │
                              ▼
                      F-beta Score (β=1.0, i.e. F1)
                      P = 1 - (|H| / |T|)    (Object Precision)
                      R = |matched_V| / |V|   (Visual Recall)
                      VTAS = (1+β²)·P·R / (β²·P + R)
```

### Key Design Decisions Made During Development

| Decision | Rationale |
|---|---|
| **Phrase-based MiniLM encoding** (`"a man"` instead of `"man"`) | Bare words gave poor similarity (man↔person = 0.45). Phrases push it to 0.72, crossing the 0.65 threshold. Eliminated the need for a hardcoded synonym dictionary. |
| **F-beta instead of `Recall × (1 - HallucinationRate)`** | User explicitly rejected hardcoded weights: *"i m not in favor of using harcoded weights without logic."* F-beta is the standard IR harmonic mean with a single mathematically interpretable parameter β. |
| **CLIP fallback threshold τ₂ = 0.22** | Empirically calibrated. CLIP similarity for true scene concepts (bedroom, fireplace) clusters around 0.25-0.30. Random noise sits around 0.15-0.18. The 0.22 threshold separates them. |
| **STOP_NOUNS list** | Words like "room", "bedroom", "kitchen", "sky", "light" are filtered before they even reach the pipeline because DETR cannot detect environments. Prevents false hallucination flags. |
| **Prefix stripping** | VLMs echo prompt artifacts ("detailed caption the dog..."). `_strip_prefix()` removes these before noun extraction so "caption" doesn't appear as a hallucinated noun. |

---

## 3. File Structure & Code Locations

```
vtas-metric/
├── src/
│   ├── vtas.py                    # Core evaluator (orchestrates all 4 modules)
│   ├── visual_grounding.py        # DETR object detection
│   ├── linguistic_extraction.py   # spaCy noun extraction + prefix stripping
│   ├── semantic_bridge.py         # MiniLM phrase-based similarity matching
│   ├── clip_fallback.py           # CLIP Tier 2 scene verification
│   ├── run_evaluation.py          # Standalone Kaggle runner script
│   └── generate_illustrations.py  # Infographic generator (matplotlib, headless)
├── tests/
│   ├── __init__.py
│   └── test_logic.py              # Unit tests (spaCy not installed locally)
├── assets/
│   ├── best_example.png           # Infographic: perfect VTAS=1.00 case
│   ├── hallucination_example.png  # Infographic: VTAS=0.00 living room
│   ├── limitation_semantic_collapse.png  # Motorcycle example for limitations
│   └── limitation_scene_penalty.png     # Living room example for limitations
├── docs/
│   └── architecture.md
├── requirements.txt
├── LICENSE                        # MIT License
├── CITATION.cff                   # GitHub auto-citation (BibTeX)
└── README.md                      # Fully documented with badges
```

### Results & Logs Location
```
VTAS_Evaluation_Metric/
├── Logs and results/
│   └── Rectified_VTAS_Extracted/
│       └── vtas_output/
│           ├── vtas_results.json          # Full per-image results (100 images)
│           └── infographics/              # 10 best + 10 worst infographics
```

---

## 4. Evaluation Results (v1.2, 100 COCO images, BLIP base)

| Metric | Value |
|---|---|
| **Average VTAS Score** | 0.566 (56.6%) |
| **Previous v1.0 Score** | 0.106 (10.6%) — before prefix stripping & CLIP |
| **Images scoring 1.00** | ~30% |
| **Images scoring 0.00** | ~25% (mostly scene-level captions) |

### Key Observations from Results
- **"man" and "people" eliminated from hallucinations list** — phrase-based encoding fixed this.
- **"fireplace", "window" rescued by CLIP** — Tier 2 working correctly.
- **Scene-only captions (kitchen, bedroom) still score 0.00** — known limitation, by design.

---

## 5. Known Limitations (Documented in README)

### Limitation 1: Semantic Collapse
**Example:** Image #78 — *"the man is riding a motorcycle on the road with people watching"*
- DETR found: `{'person', 'motorcycle'}`
- Both `"man"` (the rider) and `"people"` (spectators) matched to the single label `"person"`.
- VTAS gave 1.00 but it shouldn't — these are semantically distinct groups.
- **Root Cause:** DETR returns unique category labels, not instance counts. VTAS uses set-based matching.

### Limitation 2: Scene Penalty
**Example:** Image #0 — *"the living room has a yellow wall and a black fireplace"*
- DETR found 9 objects: `{'clock', 'tv', 'dining table', 'refrigerator', 'person', 'vase', 'potted plant', 'bottle', 'chair'}`
- Caption mentioned zero of these → Recall = 0/9 = 0.0 → F1 = 0.00
- **Root Cause:** F-beta harmonic mean collapses to 0 when Recall = 0. VTAS is an object-listing metric, not a scene-understanding metric.

---

## 6. arXiv Account Status

| Field | Value |
|---|---|
| **Username** | `ritanshu_prasad` |
| **Email** | `ritans59_soe@jnu.ac.in` |
| **Affiliation** | Independent Researcher |
| **Default Category** | `cs.AI` (should submit under `cs.CV`) |
| **Country** | India |
| **Submission Limit** | 2 papers per calendar month |
| **Endorsement** | Likely auto-granted via `.ac.in` domain |

> **Important:** The JNU email is only for arXiv account verification. The actual PDF should list the user's personal Gmail for correspondence. The user should add Gmail as a secondary email in arXiv settings as backup for when the JNU email expires.

---

## 7. Publication Roadmap (Pre-Submission Checklist)

### Phase 1: Scale Up Evaluation (1-2 days on Kaggle)
- [ ] Re-run `run_evaluation.py` on **1,000 COCO images** (currently only 100).
- [ ] Optionally test a second VLM (e.g., LLaVA or fine-tuned BLIP) to show VTAS can rank models.

### Phase 2: Baseline Comparisons (2-3 days)
- [ ] Write `run_baselines.py` to calculate **CLIPScore** for the same 1,000 images/captions.
- [ ] Calculate **BLEU / METEOR** using `pycocoevalcap` against COCO reference captions.
- [ ] Generate a **scatter plot** (VTAS vs CLIPScore) highlighting hallucination outliers.
- [ ] Compute **Pearson/Spearman correlation** between VTAS and CLIPScore.

### Phase 3: Ablation Study (1 day)
- [ ] Run VTAS with **Tier 2 (CLIP) disabled** — prove the two-tier system is necessary.
- [ ] Run VTAS with **bare-word encoding** instead of phrase-based — prove phrase encoding matters.
- [ ] Create an ablation table comparing all variants.

### Phase 4: Write the Paper (3-5 days on Overleaf)
- [ ] Use CVPR or IEEE dual-column LaTeX template.
- [ ] Sections: Abstract, Introduction, Related Work, Methodology, Experiments, Limitations, Future Work.
- [ ] Include the motorcycle and living room infographics as qualitative figures.
- [ ] Include a methodology flowchart diagram.
- [ ] Put the user's Gmail on the PDF as correspondence email.

### Suggested Paper Title
> *"VTAS: An Interpretable, Reference-Free Metric for Detecting Object Hallucinations in Image Captions"*

### Positioning Strategy
- **Do NOT** claim VTAS replaces CLIPScore or BLEU.
- **DO** position it as a complementary, interpretable, reference-free alternative.
- **DO** honestly document limitations and propose future work.
- Target: **Short paper (4-5 pages)** — modest, honest, and clean.

---

## 8. User Preferences & Constraints

| Preference | Detail |
|---|---|
| **No hardcoded weights** | User rejected arbitrary alpha/beta weights. All math must have formal justification. |
| **Decoupled Compute** | All logic in GitHub `.py` files. Kaggle notebook only clones and runs. Zero logic in notebook cells. |
| **Headless Kaggle** | No image visualization in Kaggle interactive window. Save everything to output folder. |
| **Prefix stripping** | The word "caption" must be removed from descriptions before processing — future users won't have prompt artifacts in their text. |
| **No lab email** | User does not want to use their current lab/institution email for arXiv. Using JNU alumni email instead. |
| **Kaggle Python 3.12** | Don't pin old packages. Use `datasets>=2.16.0`, `torchao>=0.16.0`. |
| **matplotlib headless** | `generate_illustrations.py` must use `matplotlib.use("Agg")`. |

---

## 9. Related Work (For the Paper's Literature Review)

| Metric | Year | Key Limitation VTAS Addresses |
|---|---|---|
| BLEU | 2002 | Purely lexical n-gram overlap. Punishes synonyms. |
| METEOR | 2005 | Uses finite WordNet synonym tables. Still text-to-text. |
| CIDEr | 2015 | Dependent on human reference consensus. Cannot detect hallucinations. |
| CHAIR | 2018 | Requires expensive human-annotated object labels per image. |
| CLIPScore | 2021 | Black-box single number. No interpretability. |
| FAITHScore | 2023 | Uses LLMs to decompose captions — more sophisticated but expensive. |
| ALOHa | 2023 | Uses object detectors similarly. Key comparison target. |
| POPE | 2023 | Polling-based probing. Different approach. |

---

## 10. Git History Summary

| Commit | Description |
|---|---|
| Initial commits | Base VTAS v1.0 (DETR + spaCy + MiniLM, naive scoring) |
| Two-Tier refactor | Added CLIP fallback, phrase-based encoding, F-beta math |
| Prefix stripping | Added `_strip_prefix()` and expanded `STOP_NOUNS` |
| `run_evaluation.py` | Decoupled Kaggle runner |
| Test fix | Fixed null-byte corruption in `tests/__init__.py` |
| README overhaul | Added parameter flaws, pictorial demos, limitations, future work |
| Public release | MIT License, CITATION.cff, shields.io badges |

---

## 11. Quick-Start for the New Chat

To continue this project in a new chat, say something like:

> *"I want to continue working on my VTAS arXiv preprint. Please read the handoff document at `c:\Users\prasa\Ritanshu\IITM\Computer Vision\Projects\VTAS_Evaluation_Metric\vtas-metric\docs\VTAS_HANDOFF.md` and then let's start Phase 1 (scaling up to 1,000 images and writing the baseline comparison script)."*

