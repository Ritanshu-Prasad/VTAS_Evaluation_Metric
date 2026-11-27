# VTAS — Visual-Truth Alignment Score

[![PyPI version](https://badge.fury.io/py/vtas-metric.svg)](https://badge.fury.io/py/vtas-metric)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**A white-box, reference-free evaluation metric for image captioning that grounds generated text directly against the visual content of the image.**

> *"Does the model see what it says it sees?"* — VTAS answers this question without needing human reference captions.

---

## Installation

```bash
pip install vtas-metric
vtas download        # one-time: fetches the spaCy model en_core_web_sm
```

Requires Python 3.10+. The first run also downloads the DETR, MiniLM and CLIP weights (~800 MB) from Hugging Face. To reproduce the paper experiments, use `pip install "vtas-metric[experiments]"`.

> **Heads-up:** VTAS 1.x is in maintenance mode. VTAS 2.0 is coming (new claim parser, calibrated verifiers, typed results); `VTASEvaluator` already emits a `DeprecationWarning`, and 1.x scores will not be comparable with 2.0 scores.

**Command line:**
```bash
vtas score path/to/image.jpg "A man throws a frisbee to his dog in the park"
```

**Python:**
```python
from vtas import VTASEvaluator

evaluator = VTASEvaluator()

result = evaluator.score(
    image_path="path/to/image.jpg",
    caption="A man throws a frisbee to his dog in the park"
)

print(f"VTAS Score (F1): {result['vtas_score']:.4f}")
print(f"Object Precision: {result['precision']:.4f}")
print(f"Visual Recall: {result['recall']:.4f}")
print(f"Matched Objects: {result['matched']}")
print(f"Hallucinated Objects: {result['hallucinated']}")
```

### Advanced Configuration (Hyperparameters)
You can heavily customize how strict VTAS is by tuning its hyperparameters during initialization:

```python
evaluator = VTASEvaluator(
    detection_confidence=0.7,  # How confident DETR must be to register a visual object
    similarity_threshold=0.65, # The MiniLM Cosine distance required to match text to boxes
    clip_threshold=0.22,       # The CLIP Cosine distance required to 'rescue' a mass noun
    beta=1.0                   # F-beta weighting (beta < 1 prioritizes Precision/Hallucinations)
)
```

---

## Why Does This Metric Exist?

Image captioning models generate natural language descriptions of images. To measure how "good" a generated caption is, the research community relies on a set of standard metrics. However, every single one of them shares a critical architectural flaw: **they are blind to the image.**

They compare the **generated text** against **human-written reference text**, but never verify whether the generated text actually describes what is physically present in the image. This creates three categories of failure:

### The CLIPScore Hallucination Problem
Modern reference-free metrics like **CLIPScore** act as a visual "bag of words." If a model hallucinates an object, CLIPScore simply averages it out and ignores the error, giving falsely high scores to terrible captions. 

> **Preliminary results.** The validation-study figures reported for VTAS (including the 18.8% hallucination rate and the r = 0.30 correlation with CLIPScore) come from an exploratory run that has **not** been checked against ground-truth hallucination labels (e.g. CHAIR on COCO annotations). Treat them as preliminary; a ground-truth evaluation is in progress for VTAS 2.0.

As shown in our exploratory 1,000-image COCO study below, VTAS actively punishes the "Hallucination Outliers" (Red Dots) that CLIPScore blindly approves:

<img src="assets/vtas_vs_clipscore.png" alt="VTAS vs CLIPScore Scatter Plot" width="600"/>

### The Problem with Legacy Metrics

#### 1. BLEU (Bilingual Evaluation Understudy)
BLEU measures n-gram precision — it counts how many word chunks in the generated caption appear in the human reference.

**The Flaw:** It is purely lexical. It has no understanding of meaning.

| Generated Caption | Reference Caption | BLEU-4 Score | Correct? |
|---|---|---|---|
| *"A man relaxes on a sofa"* | *"A person sits on a couch"* | **~0.0** (no 4-gram overlap) | ✅ Yes — semantically identical |
| *"A person sits on a couch near a dog"* | *"A person sits on a couch"* | **High** (strong overlap) | ❌ Maybe — was there actually a dog? BLEU doesn't check. |

> BLEU punishes correct captions that use synonyms, and rewards captions that copy reference words — even if the referenced objects are hallucinated.

#### 2. METEOR (Metric for Evaluation of Translation with Explicit ORdering)
METEOR improves on BLEU by adding stemming ("running" → "run") and synonym matching via external WordNet tables.

**The Flaw:** It still compares text-to-text. It relies on pre-built synonym tables that are finite, language-specific, and cannot cover domain-specific vocabulary. It never verifies against the image.

#### 3. CIDEr (Consensus-based Image Description Evaluation)
CIDEr uses TF-IDF weighting to reward captions that mention rare, image-specific words (e.g., "frisbee" scores higher than "the").

**The Flaw:** It is entirely dependent on the consensus of human reference captions. If the reference captions are biased, incomplete, or inconsistent, CIDEr scores become unreliable. It cannot detect hallucinations if the hallucinated word happens to appear in a reference.

#### 4. CLIPScore
CLIPScore is the current state-of-the-art reference-free metric. It embeds both the image and the text into a shared vector space using CLIP and computes cosine similarity.

**The Flaw:** It is a **black box**. It outputs a single similarity score but provides zero interpretability. If the score is low, there is no way to determine whether the model missed an object, hallucinated one, got a color wrong, or made a spatial error.

#### 5. CHAIR (Caption Hallucination Assessment with Image Relevance)
CHAIR was specifically designed to detect object hallucinations by comparing generated nouns against ground-truth object annotations.

**The Flaw:** It requires **expensive, human-annotated object labels** for every image. It cannot run on raw, unannotated images, making it impractical for real-world deployment at scale.

---

### The Gap

| Capability | BLEU | METEOR | CIDEr | CLIPScore | CHAIR |
|---|:---:|:---:|:---:|:---:|:---:|
| Reference-Free | ❌ | ❌ | ❌ | ✅ | ❌ |
| Detects Hallucinations | ❌ | ❌ | ❌ | Weak | ✅ |
| Handles Synonyms | ❌ | Partial | ❌ | ✅ | ❌ |
| White-Box / Interpretable | ✅ | ✅ | ✅ | ❌ | ✅ |
| Fully Automated (No Human Labels) | ✅ | ✅ | ✅ | ✅ | ❌ |

**No single existing metric is simultaneously: reference-free, hallucination-aware, synonym-robust, interpretable, AND fully automated.**

VTAS is engineered to fill that gap.

---

## What is VTAS?

VTAS (Visual-Truth Alignment Score) is a **Two-Tier Compound AI System** that evaluates image captions against physical reality:

1. **Tier 1 (Object Grounding):** An Object Detector (DETR) extracts physical objects. An NLP Parser (spaCy) extracts caption nouns. A Semantic Bridge (MiniLM, phrase-based) resolves synonyms.
2. **Tier 2 (Scene/Context Grounding):** Unmatched nouns (e.g., "bedroom", "outside", or undetected objects) are verified globally against the image using CLIP zero-shot classification.

The final **VTAS Score** uses the mathematically rigorous **F-beta Score** to balance **Object Precision** (absence of hallucinations) and **Visual Recall** (detecting visible objects) into a single metric [0, 1].

### VTAS Properties
| Property | Value |
|---|---|
| Reference-Free | ✅ No human captions needed |
| Detects Hallucinations | ✅ Explicit precision penalty for invented objects |
| Handles Synonyms | ✅ Phrase-based semantic embeddings |
| Scene Aware | ✅ Tier 2 CLIP validates complex environments |
| Interpretable | ✅ Pinpoints exactly which word failed |

---

## How the Score Is Computed

### Pipeline

```
image ──► [1] DETR ───────────────► detections D = [{label, area, ...}]
caption ─► [2] spaCy ─────────────► claims T = [{text, is_plural}]
                       │
        D, T ─► [3] MiniLM + Hungarian matching ──► matched / hallucinated(T1) / missed
                       │
 hallucinated(T1) ─► [4] CLIP (whole image) ──► rescued (grounded) / hallucinated
                       │
              [5] scoring: precision, saliency-weighted recall, F-beta
```

| Stage | Module | Model | Role |
|---|---|---|---|
| 1 | `visual_grounding.py` | `facebook/detr-resnet-50` | Detect COCO objects above `detection_confidence` (0.7); each detection carries a relative bounding-box `area`. |
| 2 | `linguistic_extraction.py` | spaCy `en_core_web_sm` | Strip prompt prefixes (e.g. "detailed caption"), take NOUN/PROPN tokens, lemmatise, drop generic nouns (`STOP_NOUNS`), flag plurals (`NNS`/`NNPS` or a numeral child). |
| 3 | `semantic_bridge.py` | `all-MiniLM-L6-v2` | **Tier 1.** Embed `"a <noun>"` / `"a <label>"` phrases; a fixed ontology first maps person words (man, woman, kid, ...) to `person`; one-to-one assignment by the Hungarian algorithm; a pair counts only if cosine similarity ≥ τ₁. |
| 4 | `clip_fallback.py` | `openai/clip-vit-base-patch32` | **Tier 2.** Unmatched nouns are scored against the full image; similarity ≥ τ₂ rescues the noun. |
| 5 | `vtas.py` (`score`) | none | Arithmetic below. |

Defaults: τ₁ = 0.65 (`similarity_threshold`), τ₂ = 0.22 (`clip_threshold`), β = 1.0.


### Mathematics (as implemented in `VTASEvaluator.score`)

**Notation.** `D` is the list of detections, each with an area `a(d)` (1.0 if absent). `T` is the list of caption claims (a noun repeated twice is two claims). `|·|` counts list entries.

#### Tier 1: one-to-one matching

Cosine similarities between claims and detections form a matrix; the Hungarian algorithm picks the assignment with maximum total similarity. A pair `(t, d)` is **matched** if `sim(t, d) ≥ τ₁`. Each detection can be used by at most one claim, so two nouns cannot both be grounded by a single detected object. Claims left unmatched are *Tier-1 unmatched*.

#### Over-count guard

An unmatched claim whose text equals an already-matched claim (e.g. "banana" said twice, one banana detected) is a **strict hallucination**: it is never sent to CLIP, because CLIP cannot count.

#### Tier 2: CLIP rescue

The remaining unmatched claims `U` go to CLIP. `M₂ = {u ∈ U : sim_CLIP(image, u) ≥ τ₂}` are *rescued* (grounded). Claims in `U \ M₂` plus the strict hallucinations form the hallucination list `H`.

#### Precision

```
P = 1 − |H| / |T|          (P = 1 when |T| = 0: no claims, no false claims)
```

#### Saliency-weighted recall

Each detection is weighted by its relative bounding-box area, so large objects matter more than small background ones.

```
plural exemption: if a plural claim ("men", "two dogs") matched a detection of label L,
                  the other unmatched detections of label L are not counted as missed.
recall rescue:    for each CLIP-rescued noun, take its best Tier-1 similarity over the
                  detected labels; if that similarity > 0.30, that label is removed from `missed`
                  (the noun is a weak synonym of a detected object, e.g. "woman" ~ "person").

S_total  = Σ_{d ∈ D} a(d)  +  0.10 · |M₂|      (virtual 10% area for each object CLIP found that DETR missed)
S_missed = Σ_{d ∈ missed} a(d)
R        = (S_total − S_missed) / S_total        (R = 1 when S_total = 0: nothing to miss)
```

#### F-beta

```
VTAS = (1 + β²) · P · R / (β² · P + R)           (VTAS = 0 when β²·P + R = 0)
```

β = 1 is F1. As in van Rijsbergen's F-measure, β < 1 weights **precision** more (hallucinations cost more) and β > 1 weights **recall** more. Range [0, 1].

#### Worked example (all values verified in `tests/test_scoring.py`)

Detections: person (.5), dog (.3), cup (.2). Caption nouns: man, dog, unicorn. man↔person and dog↔dog match; CLIP rejects unicorn; cup is missed.

```
P = 1 − 1/3 = 2/3        R = (1.0 − 0.2) / 1.0 = 4/5
F1 = 2·(2/3)(4/5) / (2/3 + 4/5) = 8/11 ≈ 0.727
```


### Output

`score()` returns `vtas_score`, `precision`, `recall`, `detected_objects`, `text_nouns`, `matched`, `clip_grounded`, `hallucinated`, `missed`, `similarity_matrix` and `clip_scores`. (Versions before 1.3 used `visual_recall` and `hallucination_rate`; those keys no longer exist.)



---

## Pictorial Demonstrations (VTAS in Action)

The following infographics were generated automatically during a VTAS evaluation run. They visually demonstrate how VTAS solves the flaws of legacy metrics.

### Example 1: Perfect Alignment & Synonym Bridging
Unlike BLEU, VTAS physically draws bounding boxes to verify reality. Even if the wording differs from human annotators, VTAS bridges the vocabulary gap using phrase-based semantic matching.
![Perfect Alignment](assets/best_example.png)

### Example 2: Scene Context & The "Recall Penalty"
Traditional metrics like CHAIR crash without human bounding boxes. VTAS operates autonomously.
* **Tier 2 Rescue:** Notice how words like "fireplace" (which DETR missed) are rescued by the CLIP context module!
* **Visual Recall Penalty:** This infographic was produced with an earlier VTAS (v1.2) that scored the caption 0.00 because it *described the scene* instead of listing the objects DETR detected. Since v1.3, recall is saliency-weighted and CLIP-rescued nouns earn recall credit, which softens (but does not remove) this penalty; see Limitation 2.
![Hallucination and Recall Penalty](assets/hallucination_example.png)

---

## Examples

### Example 1: Perfect Caption
**Image:** A park scene with a person, a dog, and a frisbee.

| Component | Output |
|---|---|
| **DETR detects** | `{'person', 'dog', 'frisbee'}` |
| **Caption says** | *"A man throws a frisbee to his dog"* → `{'man', 'frisbee', 'dog'}` |
| **Semantic Bridge** | man↔person (0.82 ✅), frisbee↔frisbee (1.0 ✅), dog↔dog (1.0 ✅) |
| **Precision / Recall** | 3/3 grounded = 1.0 / 3/3 objects mentioned = 1.0 |
| **VTAS (F1)** | **1.0** ✅ |

### Example 2: Synonym Robustness (BLEU would fail here)
**Image:** A living room with a person, a couch, and a TV.

| Component | Output |
|---|---|
| **DETR detects** | `{'person', 'couch', 'tv'}` |
| **Caption says** | *"A man relaxes on a sofa watching television"* → `{'man', 'sofa', 'television'}` |
| **Semantic Bridge** | man↔person (0.82 ✅), sofa↔couch (0.89 ✅), television↔tv (0.91 ✅) |
| **Precision / Recall** | 3/3 grounded = 1.0 / 3/3 objects mentioned = 1.0 |
| **VTAS (F1)** | **1.0** ✅ |

> BLEU-4 would score this near zero because no exact 4-grams match. VTAS correctly identifies it as a perfect caption.

### Example 3: Hallucination Detection
**Image:** A boy holding an umbrella standing near a cow.

| Component | Output |
|---|---|
| **DETR detects** | `{'person', 'umbrella', 'cow'}` |
| **Caption says** | *"A group of people with umbrellas and dogs"* → `{'people', 'umbrella', 'dog'}` ("group" is a filtered generic noun) |
| **Semantic Bridge** | people↔person (0.85 ✅), umbrella↔umbrella (0.97 ✅), dog↔cow (0.45 ❌, and CLIP rejects "dog") |
| **Precision** | 1 − 1/3 = 0.67 (hallucinated 'dog') |
| **Recall** | 2/3 = 0.67 (missed 'cow'; equal object areas assumed) |
| **VTAS (F1)** | **0.67** ❌ — Penalized |

> This is the kind of hallucination seen in a BLIP zero-shot baseline. Traditional metrics may still score it reasonably if the reference caption mentions "people." VTAS flags it because neither DETR nor CLIP finds a dog. (Illustrative values: real scores depend on bounding-box areas.)

---

## Known Limitations & Future Work

VTAS v1 is a strong first step, but we believe in transparent research. The following are **known architectural limitations** that we have identified through extensive evaluation and are actively working to resolve.

### Limitation 1: Semantic Collapse in the Semantic Bridge

> *Partly addressed in v1.3:* matching is now one-to-one per detected instance (Hungarian assignment), so two nouns can no longer both be grounded by a single detection. The example below was produced with the earlier set-based matcher; collapse can still occur when DETR finds one box for several people.

**Observed in:** Motorcycle example (Image #78, VTAS = 1.00)

![Semantic Collapse Example](assets/limitation_semantic_collapse.png)

**The Problem:** The caption reads *"the man is riding a motorcycle on the road with people watching."* DETR detected only `{'person', 'motorcycle'}`. The Semantic Bridge matched both `"man" → person (0.70)` and `"people" → person (0.76)`. On the surface, this appears correct — VTAS gave a perfect 1.00. But look closely:

* `"man"` refers to the **rider** (the main subject).
* `"people"` refers to the **spectators** in the background (secondary subjects).

These are two semantically distinct groups, but both collapsed onto the **single DETR label `"person"`**. VTAS treated them as the same entity. This is a fundamental limitation:

> **Early VTAS versions operated on unordered sets of labels.** It has no concept of *which* person is being referenced. If DETR detects one `"person"`, any number of human-related nouns ("man", "woman", "people", "boy") in the caption will all match it, even if they refer to different individuals. This inflates both Precision and Recall.

**Root Cause:** DETR returns *unique category labels*, not instance-level detections. Even if DETR internally draws 5 bounding boxes for 5 people, the VTAS pipeline reduces them to the single label `"person"`.

### Limitation 2: Scene-Level Descriptions Penalized by Object-Level Recall

**Observed in:** Living room example (Image #0, VTAS = 0.00)

![Scene Penalty Example](assets/limitation_scene_penalty.png)

**The Problem:** The caption reads *"the living room has a yellow wall and a black fireplace."* DETR detected 9 objects: `{'clock', 'tv', 'dining table', 'refrigerator', 'person', 'vase', 'potted plant', 'bottle', 'chair'}`. The caption mentioned none of these individual items — it described the **collective environment** instead. As a result:

* **Object Precision = 0.50** — `"fireplace"` was rescued by CLIP (Tier 2), but `"wall"` was marked as a hallucination.
* **Visual Recall = 0.00** — The caption mentioned 0 out of 9 detected objects.
* **VTAS (F1) = 0.00** — Because F1 is a harmonic mean, if either P or R is 0, the entire score collapses.

This is arguably the **most important limitation**: a VLM that writes *"the kitchen is clean and ready for us to use"* receives a VTAS of 0.00, even though that caption is a perfectly valid, human-like description of a kitchen. The model correctly identified the scene but chose to describe it holistically rather than itemizing the oven, refrigerator, sink, cups, and bowls that DETR found.

> **VTAS v1 is an object-listing metric, not a scene-understanding metric.** It heavily rewards captions that enumerate discrete objects ("there is a clock, a TV, a chair, and a bottle") and penalizes captions that describe the scene as a whole ("this is a clean, well-lit living room"). This is by design — VTAS measures *grounded object alignment* — but it means that VTAS alone cannot evaluate the full quality of a caption.

**Root Cause:** The F-beta score requires non-zero Recall. When a caption contains zero object-level nouns that overlap with DETR's detections, Recall = 0, and the harmonic mean forces the entire score to 0 regardless of Precision.

---

### Future Work

Based on the limitations above, we propose the following research directions for VTAS v2:

| Direction | Addresses | Approach |
|---|---|---|
| **Instance-Aware Grounding** | Limitation 1 | Replace set-based matching with instance-level alignment. Use DETR's bounding box coordinates to distinguish "the rider" from "the spectators". Match each caption noun to a specific detection, preventing multiple nouns from collapsing onto one label. |
| **Scene-Level Recall** | Limitation 2 | Introduce a secondary recall signal that rewards captions for correctly identifying the *type* of scene (kitchen, bedroom, park) using CLIP or a dedicated scene classifier (e.g., Places365). A caption like "the kitchen is clean" would earn partial Recall credit for correctly classifying the environment. |
| **Attribute-Level Grounding** | New capability | Extend VTAS beyond nouns to verify adjectives ("red car", "large dog"). Use visual question-answering or attribute classifiers to check if described properties (color, size, material) match the image. |
| **Spatial Awareness** | New capability | Verify spatial claims ("the dog is *next to* the person", "the cup is *on* the table") against the physical coordinates of DETR bounding boxes. |
| **Dynamic Threshold Calibration** | Both | Replace the fixed similarity thresholds (MiniLM τ=0.65, CLIP τ=0.22) with dataset-adaptive thresholds calibrated on a held-out validation split, similar to ROC-AUC optimal threshold selection. |

---


## Repository Structure

- **src/vtas/**: the PyPI-installable package (detection, extraction, semantic bridge, CLIP fallback, scoring, `vtas` CLI).
- **experiments/**: scripts used for the paper experiments (`run_evaluation.py`, `run_baselines.py`, `ablation.py`); need the `experiments` extra.
- **examples/**: tutorials such as `batch_test_vtas.py` for scoring a folder of images.
- **harness/**: evaluation harness for VTAS 2.0 (CHAIR-GT, fixed COCO calib/dev/test splits, pre-registered gates, external dataset loaders); see [harness/README.md](harness/README.md). Not part of the PyPI package.
- **tests/**: unit tests. `pytest -q` runs the fast CPU tests; `pytest -m integration` also loads the real spaCy/MiniLM models.

---

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

If you use VTAS in your research, please cite:

```bibtex
@software{prasad2026vtas,
  author    = {Prasad, Ritanshu},
  title     = {VTAS: Visual-Truth Alignment Score},
  year      = {2026},
  url       = {https://github.com/Ritanshu-Prasad/VTAS_Evaluation_Metric},
  license   = {MIT}
}
```

## Acknowledgements

- The evaluation harness (`harness/`: CHAIR-GT, fixed COCO splits, dataset loaders) was
  generated with the assistance of Claude Code and reviewed by the author.
- CHAIR metric and synonym list: Rohrbach et al., "Object Hallucination in Image Captioning" (EMNLP 2018).
  Pattern 3.6 `singularize` (CLiPS, University of Antwerp, BSD-3). See `harness/resources/NOTICE`.
- Datasets: COCO, FOIL, THumB, Flickr8k, Pascal-50S, VSR, What'sUp, AMBER and nocaps;
  sources and access notes are in `harness/datasets/`.
