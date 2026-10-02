# VTAS — Visual-Truth Alignment Score

**A white-box, reference-free evaluation metric for image captioning that grounds generated text directly against the visual content of the image.**

---

## Why Does This Metric Exist?

Image captioning models generate natural language descriptions of images. To measure how "good" a generated caption is, the research community relies on a set of standard metrics. However, every single one of them shares a critical architectural flaw: **they are blind to the image.**

They compare the **generated text** against **human-written reference text**, but never verify whether the generated text actually describes what is physically present in the image. This creates three categories of failure:

### The Problem with Existing Metrics

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

The final **VTAS Score (v1.2)** uses the mathematically rigorous **F-beta Score** to balance **Object Precision** (absence of hallucinations) and **Visual Recall** (detecting visible objects) into a single metric [0, 1].

### VTAS Properties
| Property | Value |
|---|---|
| Reference-Free | ✅ No human captions needed |
| Detects Hallucinations | ✅ Explicit precision penalty for invented objects |
| Handles Synonyms | ✅ Phrase-based semantic embeddings |
| Scene Aware | ✅ Tier 2 CLIP validates complex environments |
| Interpretable | ✅ Pinpoints exactly which word failed |

---

## Pictorial Demonstrations (VTAS in Action)

The following infographics were generated automatically during a VTAS evaluation run. They visually demonstrate how VTAS solves the flaws of legacy metrics.

### Example 1: Perfect Alignment & Synonym Bridging
Unlike BLEU, VTAS physically draws bounding boxes to verify reality. Even if the wording differs from human annotators, VTAS bridges the vocabulary gap using phrase-based semantic matching.
![Perfect Alignment](assets/best_example.png)

### Example 2: Scene Context & The "Recall Penalty"
Traditional metrics like CHAIR crash without human bounding boxes. VTAS operates autonomously.
* **Tier 2 Rescue:** Notice how words like "fireplace" (which DETR missed) are rescued by the CLIP context module!
* **Visual Recall Penalty:** This caption scored an F1 of 0.00 not because it hallucinated wildly, but because it *described the scene* instead of listing the physical objects detected by DETR. 
![Hallucination and Recall Penalty](assets/hallucination_example.png)

---

## Quick Start

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

---

## Examples

### Example 1: Perfect Caption
**Image:** A park scene with a person, a dog, and a frisbee.

| Component | Output |
|---|---|
| **DETR detects** | `{'person', 'dog', 'frisbee'}` |
| **Caption says** | *"A man throws a frisbee to his dog"* → `{'man', 'frisbee', 'dog'}` |
| **Semantic Bridge** | man↔person (0.82 ✅), frisbee↔frisbee (1.0 ✅), dog↔dog (1.0 ✅) |
| **Visual Recall** | 3/3 = 1.0 |
| **Hallucination Rate** | 0/3 = 0.0 |
| **VTAS** | **1.0** ✅ |

### Example 2: Synonym Robustness (BLEU would fail here)
**Image:** A living room with a person, a couch, and a TV.

| Component | Output |
|---|---|
| **DETR detects** | `{'person', 'couch', 'tv'}` |
| **Caption says** | *"A man relaxes on a sofa watching television"* → `{'man', 'sofa', 'television'}` |
| **Semantic Bridge** | man↔person (0.82 ✅), sofa↔couch (0.89 ✅), television↔tv (0.91 ✅) |
| **Visual Recall** | 3/3 = 1.0 |
| **Hallucination Rate** | 0/3 = 0.0 |
| **VTAS** | **1.0** ✅ |

> BLEU-4 would score this near zero because no exact 4-grams match. VTAS correctly identifies it as a perfect caption.

### Example 3: Hallucination Detection
**Image:** A boy holding an umbrella standing near a cow.

| Component | Output |
|---|---|
| **DETR detects** | `{'person', 'umbrella', 'cow'}` |
| **Caption says** | *"A group of people with umbrellas and dogs"* → `{'group', 'people', 'umbrellas', 'dogs'}` |
| **Semantic Bridge** | people↔person (0.85 ✅), umbrellas↔umbrella (0.97 ✅), group↔? (❌), dogs↔cow (0.45 ❌) |
| **Visual Recall** | 2/3 = 0.67 (missed 'cow') |
| **Hallucination Rate** | 2/4 = 0.50 (hallucinated 'group', 'dogs') |
| **VTAS** | **0.33** ❌ — Correctly penalized |

> This is a real hallucination observed during our BLIP zero-shot baseline evaluation. Traditional metrics may still score this reasonably if the reference caption mentions "people." VTAS catches it because DETR confirms there is no "group" or "dogs" in the image.

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- `torch` — PyTorch backend
- `transformers` — Hugging Face (DETR object detection)
- `sentence-transformers` — MiniLM semantic embeddings
- `spacy` — NLP noun extraction
- `Pillow` — Image processing

---

## Project Structure
```
VTAS_Evaluation_Metric/
├── src/
│   ├── vtas.py                  # Core VTAS evaluator class
│   ├── visual_grounding.py      # DETR-based object detection module
│   ├── linguistic_extraction.py # spaCy-based noun extraction module
│   └── semantic_bridge.py       # MiniLM-based semantic similarity module
├── docs/
│   └── architecture.md          # Full architectural specification & math
├── requirements.txt
├── .gitignore
└── README.md
```

---

## License
This project is currently private and under active development.
