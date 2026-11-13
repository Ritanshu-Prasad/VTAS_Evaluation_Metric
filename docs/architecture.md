# VTAS 1.3 Architecture

**VTAS (Visual-Truth Alignment Score)** is a reference-free metric for image captions. It checks the nouns a caption claims against what an object detector and CLIP find in the image, and reports precision (no invented objects), recall (visible objects mentioned) and their F-beta combination.

> **Status.** VTAS 1.3.x is in maintenance mode (bug fixes only). VTAS 2.0 is coming with a new claim parser, calibrated verifiers and a typed result object; `VTASEvaluator` will then become a thin wrapper that emits a `DeprecationWarning`, and **1.x scores will not be comparable with 2.0 scores**. Since 1.3.1, constructing `VTASEvaluator` already emits a `DeprecationWarning` to that effect. See [ROADMAP_V2.md](ROADMAP_V2.md).

All results quoted for VTAS so far are **preliminary**: they have not yet been validated against ground-truth hallucination labels (see the roadmap, Section 6).

---

## Pipeline

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

---

## Mathematics (as implemented in `VTASEvaluator.score`)

**Notation.** `D` is the list of detections, each with an area `a(d)` (1.0 if absent). `T` is the list of caption claims (a noun repeated twice is two claims). `|·|` counts list entries.

### Tier 1: one-to-one matching

Cosine similarities between claims and detections form a matrix; the Hungarian algorithm picks the assignment with maximum total similarity. A pair `(t, d)` is **matched** if `sim(t, d) ≥ τ₁`. Each detection can be used by at most one claim, so two nouns cannot both be grounded by a single detected object. Claims left unmatched are *Tier-1 unmatched*.

### Over-count guard

An unmatched claim whose text equals an already-matched claim (e.g. "banana" said twice, one banana detected) is a **strict hallucination**: it is never sent to CLIP, because CLIP cannot count.

### Tier 2: CLIP rescue

The remaining unmatched claims `U` go to CLIP. `M₂ = {u ∈ U : sim_CLIP(image, u) ≥ τ₂}` are *rescued* (grounded). Claims in `U \ M₂` plus the strict hallucinations form the hallucination list `H`.

### Precision

```
P = 1 − |H| / |T|          (P = 1 when |T| = 0: no claims, no false claims)
```

### Saliency-weighted recall

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

### F-beta

```
VTAS = (1 + β²) · P · R / (β² · P + R)           (VTAS = 0 when β²·P + R = 0)
```

β = 1 is F1. As in van Rijsbergen's F-measure, β < 1 weights **precision** more (hallucinations cost more) and β > 1 weights **recall** more. Range [0, 1].

### Worked example (all values verified in `tests/test_scoring.py`)

Detections: person (.5), dog (.3), cup (.2). Caption nouns: man, dog, unicorn. man↔person and dog↔dog match; CLIP rejects unicorn; cup is missed.

```
P = 1 − 1/3 = 2/3        R = (1.0 − 0.2) / 1.0 = 4/5
F1 = 2·(2/3)(4/5) / (2/3 + 4/5) = 8/11 ≈ 0.727
```

---

## Output

`score()` returns `vtas_score`, `precision`, `recall`, `detected_objects`, `text_nouns`, `matched`, `clip_grounded`, `hallucinated`, `missed`, `similarity_matrix` and `clip_scores`. (Versions before 1.3 used `visual_recall` and `hallucination_rate`; those keys no longer exist.)

---

## Known limitations of the 1.3 design

* **Object-nouns only.** Colours, counts, relations and scene descriptions are not verified.
* **Scene-style captions can score near zero**, because recall requires mentioning detected objects (the CLIP recall rescue softens this but does not remove it).
* **Detector vocabulary.** DETR only knows the COCO categories; everything else relies on CLIP.
* **Hand-set thresholds.** τ₁, τ₂, the 0.30 rescue similarity, the 0.10 bonus area and the generic-noun list were set by hand, not calibrated against labels. VTAS 2.0 replaces them with calibrated values.
* **Person ontology.** Person synonyms are collapsed onto `person` before embedding.
