# VTAS Rectification Plan (v1.3.0 on PyPI → VTAS 2.0)

**Status date:** 2026-11-13
**Scope:** Everything needed to take VTAS from the current PyPI release (`vtas-metric==1.3.0`) to a metric whose claims are backed by ground truth, human judgment and a reproducible package.
**Audience:** the VTAS author and anyone (human or agent) continuing this work in a later session. Read Section 1 before touching code.

---

## 0. How to use this document

- Section 1 is the audit of what is wrong today, with numbers taken from the Phase 1 run (`Preprint/Phase 1/results/vtas_output/vtas_results.json`, 1,000 COCO images, BLIP base).
- Section 2 is the non-negotiable design rules for v2.
- Section 3 is the v2 architecture with the exact runtime ML model for each component (the models that ship inside the metric).
- Section 4 is the Concept Policy (standardization of person/boy/people, counts, synonyms).
- Section 5 is the phased work plan. Every step lists the runtime ML models it needs and its done-criterion.
- Section 5A assigns a Claude model and effort level to every step (the models that build and check the work, not part of the metric).
- Section 6 is the validation plan: each claim the paper will make, the test that backs it, the dataset, the metric, the pass threshold and the baseline.
- Section 7 covers PyPI and backwards compatibility.
- Section 8 lists risks with fallbacks.
- Section 9 records what was re-verified on 2026-11-13 and what changed from the earlier chat plan.

The one rule that governs everything: **no change to the metric ships unless it improves agreement with ground truth or human judgment on the harness in Section 6. The mean VTAS score is never evidence.**

---

## 1. Audit of VTAS v1.3.0 (what must be fixed)

### 1.1 Metric-level problems (measured)

| # | Finding | Evidence | Root cause |
|---|---|---|---|
| M1 | The reported 18.8% sentence-level hallucination rate is mostly metric error | Of 215 flagged nouns: 30% are stuff/background (grass, water, building), 15% are fragments of COCO compound labels (fire ← fire hydrant, cell ← cell phone), 14% are garbled BLIP tokens (frc, col), 7% clothing/body parts. Only ~33% plausibly real. In 52% of the 188 flagged images every flag is an artifact. | Per-token noun extraction instead of noun chunks; detector covers 80 "thing" classes only; stop-list used as a patch |
| M2 | Tier 2 (CLIP) is a near pass-through | Rescues 78% of Tier 1 rejections. Rescued scores: min 0.220, median 0.250, threshold 0.22. | Raw CLIP cosine against a fixed global threshold is not a calibrated presence test |
| M3 | Recall penalizes non-exhaustive captions | 4,858 missed detections vs 1,303 matches. Image 318 "a cobbled street": P=1.0, score 0.00 (26 detections). Recall mean 0.77, 11% of images < 0.4 | Captions are not inventories; every detection counts as a miss |
| M4 | Validation is circular | Ablation compares mean VTAS (0.804 vs 0.585 vs 0.574) and calls higher better; it reverse-engineers inputs from JSON instead of re-running. Mean drifted 0.106 → 0.342 → 0.566 → 0.804 across versions, each patch kept because the mean rose. | No ground truth or human data anywhere in the loop |
| M5 | Baselines are broken | CHAIRi = 0.0 (text nouns never saved → zero denominator). BLEU/METEOR never ran (`nlg_metrics: {}`) because `merve/coco` has no image ids; `image_id == index` for all 1,000 rows. "CHAIR" in the summary is VTAS's own output. | Wrong dataset loader; missing fields |
| M6 | Hyperparameters misreported | Experiment ran τ1=0.65, τ2=0.22 (hardcoded in `run_evaluation.py`). Paper context and README state 0.60 / 0.15. | Runner overrides module defaults |
| M7 | Hidden magic constants | 0.10 virtual area per CLIP rescue, 0.30 weak-synonym cutoff, COCO ontology dict (man→person makes sim = 1.0 for 327 matches, bypassing the MiniLM "novelty"); "stop sign" counted twice (stop matched + sign CLIP-rescued) | Patches added without a ground-truth check |
| M8 | Detector dependence (author-reported) | Any object class the detector was not trained on, or any domain shift, becomes a "hallucination" | Absence of detection is treated as absence of the object |
| M9 | No spatial / environment understanding (author-reported) | on, under, in front, behind, left, right are ignored; scene words are either stop-listed or sent to the CLIP gate | Nouns-only extraction |
| M10 | No standardization of concept granularity (author-reported) | When is "boy" a "person"? How many persons are "people"? v1 answers via an ad-hoc dict and a plural flag | No written policy; no human agreement data |

### 1.2 Engineering / open-source problems

| # | Finding |
|---|---|
| E1 | Tests cannot run: `tests/__init__.py` is UTF-16 with BOM (null bytes); test file imports the pre-package module path; passes sets of strings where the bridge needs lists of dicts; several assertions compare strings against dicts and pass vacuously; the scoring math in `score()` has no test |
| E2 | `examples/batch_test_vtas.py` reads removed keys (`visual_recall`, `hallucination_rate`) and uses a stale import |
| E3 | PyPI install fails on first use: `scipy` imported but not declared; spaCy model `en_core_web_sm` never installed; placeholder `example.com` author email published; summary says "Visual-Text" not "Visual-Truth"; `python_requires>=3.8` vs README "3.10+" |
| E4 | Version drift: setup.py 1.3.0, CITATION.cff 1.2.0, docstrings/CLI banner v1.1 and v1.2 |
| E5 | Stale docs: `architecture.md` still describes VR × (1−HR) with τ=0.75 and a v2 = CLIP-only roadmap; README duplicates Installation/Quick Start, old project tree, pre-v1.3 "scene captions score 0.00" claim |
| E6 | No GPU path (DETR and CLIP always on CPU), no batching, no detection caching, `print` instead of logging, no `pyproject.toml`, no CI, no CONTRIBUTING/CHANGELOG, experiment-only deps (peft, datasets, torchao) in the library requirements |
| E7 | Live PyPI token in `.pypirc` inside the working tree (gitignored, never committed, but should move out and be replaced by trusted publishing) |

---

## 2. Design rules for VTAS 2.0

1. **Claims, not nouns.** The caption is parsed into typed claims: object, count, attribute, relation, scene. Each claim receives its own verdict and evidence.
2. **Three-way verdicts.** Every claim is `SUPPORTED`, `CONTRADICTED`, or `UNVERIFIABLE`. Only `CONTRADICTED` is a hallucination. `UNVERIFIABLE` is excluded from precision and reported as reduced coverage. This is the fix for M8.
3. **Contradiction needs independent evidence.** A claim is `CONTRADICTED` only when at least two verifiers with measured recall for that concept agree it is absent. A verifier that cannot see a class cannot testify to its absence.
4. **Recall uses an independent inventory.** Coverage is computed against objects found by a fixed-vocabulary panoptic model, never against the caption's own grounding queries (otherwise recall is 1.0 by construction).
5. **No hand-set thresholds.** Every cutoff is fitted on a calibration split at a stated false-positive rate and reported with its ROC curve.
6. **Components are reported separately.** VTAS-P (claim faithfulness), VTAS-R (saliency-weighted coverage), VTAS-A (attributes), VTAS-C (counts), VTAS-S (spatial/scene), plus Coverage. A headline F-beta may be offered, with beta chosen from human preference data (THumB), never by hand.
7. **A written, versioned Concept Policy** (Section 4) governs synonymy, specificity and counts. Code reads the policy file; the policy is not hidden in code.
8. **Everything reproducible.** Official COCO files with real image ids, pinned model revisions, seeds, and a harness that re-runs end-to-end.

---

## 3. VTAS 2.0 architecture and the runtime ML model for each component

Two install profiles ship: **light** (runs on a free Kaggle T4 / 16 GB) and **full** (adds the VLM probe, depth, and large backbones; tested on the RTX PRO 6000).

| Component | Role | Model (Hugging Face id) | Size / VRAM (approx.) | Profile | Notes |
|---|---|---|---|---|---|
| C1 Claim parser | Noun chunks with compound merging, numerals, adjectives, prepositional relations, scene words | spaCy `en_core_web_trf` (default) / `en_core_web_sm` (light) | 440 MB / CPU | both | Optional LLM extraction mode (`Qwen/Qwen2.5-1.5B-Instruct`) for long VLM captions, as ALOHa does; off by default for reproducibility |
| C2 Panoptic inventory | Things + stuff with pixel masks; the recall inventory; saliency by mask area; instance counts | `facebook/mask2former-swin-large-coco-panoptic` (full) / `facebook/mask2former-swin-tiny-coco-panoptic` (light) | 216 M / ~6 GB; 47 M / ~2 GB | both | COCO panoptic = 80 things + 53 stuff (133 classes). **License caveat:** Mask2Former weights are non-commercial; the research release is fine, but document it. Apache fallback: Detectron2 Panoptic-FPN R50 |
| C3 Open-vocabulary grounder | Verifies object claims outside the 133-class vocabulary; returns boxes + scores | `google/owlv2-base-patch16-ensemble` (default); `IDEA-Research/grounding-dino-tiny` (second opinion) | 150 M / ~3 GB; 172 M / ~3 GB | both (one grounder in light, two in full) | Pure `transformers`, no CUDA compile. Scores are uncalibrated → calibration step S1.6 is mandatory |
| C4 Attribute verifier | Checks colour/material/state on the crop of the matched region | `google/siglip-so400m-patch14-384` zero-shot over an attribute set (full); `openai/clip-vit-base-patch32` (light) | 878 M / ~4 GB; 151 M / ~1 GB | both | Open-vocabulary detectors are unreliable for attributes (Bianchi et al., CVPR 2024), so attributes are never judged from the detector score |
| C5 VLM yes/no probe | Third vote for contradiction; also used for `UNVERIFIABLE` triage | `Qwen/Qwen2-VL-2B-Instruct` (default full); `HuggingFaceTB/SmolVLM-Instruct` (lighter) | 2.2 B / ~6 GB bf16 | full only | VLMs hallucinate too; it is one vote of three, never the sole judge |
| C6 Depth estimator | Depth ordering for in front of / behind | `depth-anything/Depth-Anything-V2-Small-hf` | 25 M / ~1 GB | full | Apache-2.0 (Small only; Base/Large are CC-BY-NC) |
| C7 Scene classifier | Verifies kitchen/beach/outdoors-type scene claims | Zero-shot SigLIP/CLIP over the 365 Places365 labels + stuff-mask evidence from C2 | shares C4 | both | No maintained `transformers` Places365 head exists; zero-shot with calibrated softmax is the reproducible choice |
| C8 Relation geometry | on, under, above, below, in, left, right, near, in front, behind | No model; functions over C2 masks and C6 depth with fitted fuzzy membership | CPU | both | Membership curves are fitted on VSR / What'sUp training splits, not hand-set |
| C9 Calibration module | Fits all thresholds at a target FPR on the calibration split; stores them with the policy version | scikit-learn (isotonic / Platt) | CPU | both | Outputs `calibration.json` that the paper reports verbatim |
| C10 Legacy v1 | Reproduces the published 1.3.0 numbers | DETR R50 + MiniLM + CLIP B/32 + spaCy sm (as today) | ~1 GB | `vtas[legacy]` | Kept as `vtas.legacy`, frozen |

**Captioners used for evaluation (not part of the metric):**

| Model | HF id | Why |
|---|---|---|
| BLIP base | `Salesforce/blip-image-captioning-base` | Continuity with v1 results |
| BLIP-2 OPT-2.7B | `Salesforce/blip2-opt-2.7b` | Stronger classic captioner |
| GIT base COCO | `microsoft/git-base-coco` | Different architecture family |
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | Long, hallucination-prone VLM captions |
| Qwen2-VL-2B / 7B | `Qwen/Qwen2-VL-2B-Instruct` | Modern VLM; also our probe (use the 7B as captioner to avoid judging itself with itself) |
| InstructBLIP | `Salesforce/instructblip-vicuna-7b` | Sixth system for ranking tests |

**Baseline metrics to implement or wrap:** CHAIR with ground truth (own implementation on COCO instances + standard synonym list), CLIPScore and RefCLIPScore (ViT-B/32, "A photo depicts" prompt, w = 2.5), PAC-S (`aimagelab/pacscore`), ALOHa (`pip install git+https://github.com/DavidMChan/aloha.git`), FaithScore (if its LLM dependency can be run locally), BLEU-4 / METEOR / CIDEr / SPICE via `pycocoevalcap`.

---

## 4. Concept Policy v1 (standardization)

Stored as `vtas/policy/concept_policy_v1.yaml`, versioned, and loaded at runtime. The paper states the policy version used.

### 4.1 Specific vs general terms
- A **general** term is `SUPPORTED` by any more specific visible instance in its hyponym set: "person" ← boy, girl, man, woman, player, rider, child, baby.
- A **specific** term is `SUPPORTED` at the parent level when the parent is visible and gets an **optional soft check** for its distinguishing attribute. "Boy" ← person detected + age/gender probe. In the **standard** profile a failed soft check marks the claim `UNVERIFIABLE`, never `CONTRADICTED`; in **strict** it becomes `CONTRADICTED` only if both the attribute verifier and the VLM probe disagree.
- Hyponym sets come from WordNet plus a reviewed override file; MiniLM similarity is no longer used for synonymy.

### 4.2 Counts and collective nouns (claims are count intervals)

| Expression | Interval |
|---|---|
| "a X", "one X", "the X" | [1, ∞) in lenient; [1, 1] in strict; [1, 2] in standard (one salient instance plus tolerance for cropped duplicates) |
| "two X", "three X" | exact ±0 (standard), ±1 (lenient) |
| "X-s" (bare plural), "some X", "a few X" | [2, ∞) |
| "a couple of X", "a pair of X" | [2, 2] (standard), [2, 3] (lenient) |
| "several X" | [3, ∞) |
| "a group of X" | [3, ∞) |
| "a crowd", "crowds", "many people" | [6, ∞) |

Intervals for "several", "group", "crowd", "many" are **fitted** on human count judgments collected on the golden set (Section 5, S0.5), then frozen in the policy file. Each noun phrase claims a set of instances; two phrases cannot claim the same instance (Hungarian assignment over instances, not labels), which ends the man/people collapse.

### 4.3 Strictness profiles
`strict`, `standard` (default), `lenient`. They change: synonym leeway, count tolerance, treatment of soft attribute checks, and the FPR used for calibration (2%, 5%, 10%).

### 4.4 Out-of-vocabulary handling
If a claim's head noun is outside every verifier's trusted vocabulary (trusted = per-class recall ≥ 0.6 on the calibration split), the claim is `UNVERIFIABLE` by construction and lowers Coverage. This is the formal answer to M8.

---

## 5. Phased work plan (runtime ML models per step; Claude assignment in Section 5A)

Timeline assumes one person, Kaggle for light runs and the RTX PRO 6000 for full runs.

### Phase 0 — Stop the bleeding and build the harness (Week 1)

| Step | Work | Runtime ML models used | Done when |
|---|---|---|---|
| S0.1 | Fix tests (UTF-8 `__init__.py`, package imports, dict inputs, add pure-math tests for `score()` with a mocked pipeline) | none | `pytest -q` green on CPU in < 30 s |
| S0.2 | Fix packaging: add `scipy`, auto-install spaCy model (post-install hook or `vtas download`), real author email, correct summary, `python_requires>=3.10`, `pyproject.toml`, single version source | none | fresh venv: `pip install vtas-metric && vtas score img.jpg "a dog"` works |
| S0.3 | Publish **1.3.1** (bugfix only) with a `DeprecationWarning` pointing to 2.0; move `.pypirc` out of the tree; set up PyPI trusted publishing via GitHub Actions | none | 1.3.1 live; token revoked and rotated |
| S0.4 | Docs: rewrite `architecture.md` to match 1.3 maths; dedupe README; add a "Preliminary results" banner over the 18.8% and r = 0.30 claims | none | README has no contradictions with code |
| S0.5 | Build the **ground-truth harness**: download official COCO val2014 images + `instances_val2014.json` + `captions_val2014.json`; implement true CHAIR; fixed splits: calib (1k), dev (1k), test (3k) by image id; store split files in repo | none | CHAIR-GT on BLIP captions reproduces the published ballpark for BLIP (CHAIRs ≈ 10–15%) |
| S0.6 | Build the **golden set**: 300 captions (100 BLIP, 100 LLaVA, 100 human references) hand-labelled per claim (object / count / attribute / relation / scene; verdict; note). Two annotators + adjudication. | none | Cohen's kappa ≥ 0.7 on verdicts |
| S0.7 | Download external sets: FOIL, THumB 1.0, Flickr8k-Expert/CF, Pascal-50S, VSR, What'sUp, AMBER, nocaps val; write loaders | none | each loader has a unit test with a checksum |

### Phase 1 — Core v2 (Weeks 2–4)

| Step | Work | Runtime ML models used | Done when |
|---|---|---|---|
| S1.1 | Claim parser C1: noun chunks, compound merge, numerals → count intervals, adjectives → attribute claims, prepositions → relation triples, scene lexicon | spaCy trf / sm | ≥ 95% claim-extraction agreement with golden-set claim lists |
| S1.2 | Panoptic inventory C2 with per-class recall table measured on calib split against COCO panoptic GT | Mask2Former (tiny + large) | recall table saved; stuff classes resolve M1's 30% bucket |
| S1.3 | Open-vocab grounder C3 for OOV object claims; second grounder in full profile | OWLv2; Grounding DINO tiny | per-query scores stored with boxes |
| S1.4 | Three-way verdict engine (Section 2 rules 2–3) with instance-level Hungarian assignment | C2 + C3 (+ C5 in full) | golden-set object-verdict accuracy ≥ 0.9 |
| S1.5 | Saliency-weighted recall from masks; component scores VTAS-P / VTAS-R / Coverage; structured JSON evidence output | C2 | output schema documented and validated with `pydantic` |
| S1.6 | Calibration C9: fit grounder / detector / VLM thresholds at FPR 2/5/10% on calib split; write `calibration.json` | all above | ROC curves plotted; thresholds no longer appear as literals in code |
| S1.7 | Run harness tests T1–T6 (Section 6) on dev split; iterate; final numbers on test split once | all above | **Gate A** and **Gate D** pass |

### Phase 2 — Spatial, scene, attributes, counts (Weeks 4–6)

| Step | Work | Runtime ML models used | Done when |
|---|---|---|---|
| S2.1 | Relation geometry C8 with fuzzy membership fitted on VSR train; depth ordering via C6 | Depth Anything V2 Small | VSR dev accuracy reported per relation |
| S2.2 | Scene verifier C7 | SigLIP / CLIP zero-shot over Places365 labels | scene-claim accuracy on golden set ≥ 0.85 |
| S2.3 | Attribute verifier C4 on crops | SigLIP so400m / CLIP B/32 | AMBER attribute-dimension accuracy reported |
| S2.4 | Count verifier VTAS-C with policy intervals; fit intervals for several/group/crowd on golden-set human counts | C2 instance masks | count-claim accuracy ≥ 0.85 on golden set |
| S2.5 | Strictness profiles wired to policy + calibration | none | three profiles produce monotone scores on perturbation suite |
| S2.6 | Run T7–T12 | all | **Gates E and F** pass |

### Phase 3 — Breadth, baselines, ablations (Weeks 6–7)

| Step | Work | Runtime ML models used | Done when |
|---|---|---|---|
| S3.1 | Generate captions from six captioners on the 3k test split + nocaps val | captioner table above | caption files versioned with model revision hashes |
| S3.2 | Implement/wrap all baselines | CLIP B/32, PAC-S, ALOHa, pycocoevalcap | baseline numbers reproduce their papers on a sanity subset |
| S3.3 | Ablations on the harness (not on mean score): panoptic vs DETR, chunks vs tokens, calibrated vs fixed, each verifier removed, each grounder alone, light vs full | all | ablation table with bootstrap 95% CIs |
| S3.4 | OOD suite: nocaps, sketches/clipart subset, night images, 100 medical/satellite images with reference captions → false-alarm rate and coverage | all | **Gate G** pass |
| S3.5 | Run T13–T24 | all | **Gates B and C** pass |

### Phase 4 — Paper and release (Weeks 8–9)

| Step | Work | Done when |
|---|---|---|
| S4.1 | Paper: position as "annotation-free, claim-level CHAIR with coverage-aware verdicts"; related work covers CHAIR, ALOHa, OpenCHAIR, AMBER, POPE, FaithScore, CLIPScore/PAC-S, THumB; report policy version and calibration file | internal review checklist in Section 6.5 all ticked |
| S4.2 | Release `vtas-metric==2.0.0`: `pyproject.toml`, extras `[light]`, `[full]`, `[legacy]`, CLI (`vtas score`, `vtas eval --jsonl`, `vtas calibrate`), GPU + batching + caching, CI matrix (3.10–3.12, CPU), pinned model revisions, CHANGELOG, CONTRIBUTING, Zenodo DOI, benchmark JSON | `pip install vtas-metric[light]` → score in one command on a T4 |
| S4.3 | arXiv cs.CV submission + workshop target (CVPR/ICCV workshop on evaluation or hallucination; or ACL/EMNLP Findings) | submitted |

### Phase 5 — Adoption (ongoing)
lmms-eval / VLMEvalKit integration PRs, public leaderboard JSON, policy-file contributions via PRs, demo notebook, issue triage. This phase is what turns a 9 into a 10 and cannot be scheduled.

---

## 5A. Claude model and effort assignment per step

These are the Claude models that do the engineering, analysis and writing. None of them is part of the metric at runtime, and no Claude model is ever used as evidence for a metric claim (no Claude-as-judge numbers in the paper).

**Model ids:** Opus 5.5 `claude-opus-5-5`, Sonnet 5.5 `claude-sonnet-5-5`, Haiku 5.5 `claude-haiku-5-5`. Effort levels are low, medium, high, xhigh, max.

**Selection rule:** Haiku for mechanical, well-specified work with a test that catches mistakes. Sonnet for most implementation. Opus where a subtle error would silently invalidate results, which means the evaluation harness, the verdict logic, statistics, spatial geometry and the paper. Effort goes up with how costly an undetected mistake is, not with how long the task is.

| Step | Claude model | Effort | Why |
|---|---|---|---|
| S0.1 Fix tests | Sonnet 5.5 | medium | Well-scoped repair; the math fixtures need care |
| S0.2 Packaging | Sonnet 5.5 | medium | pyproject, extras, post-install spaCy hook |
| S0.3 Release 1.3.1 | Sonnet 5.5 | medium | Workflow for trusted publishing; the token rotation itself is manual by the author |
| S0.4 Docs cleanup | Sonnet 5.5 | medium | architecture.md must match the real maths; Haiku low for the README dedupe pass |
| S0.5 Ground-truth harness (CHAIR-GT) | Opus 5.5 | high | Foundation of every later claim; an error here poisons everything |
| S0.6 Golden set | Sonnet 5.5 | high | Drafts the rubric and pre-labels. Humans produce the labels used for kappa; Claude pre-labels are reported separately and never counted as annotators |
| S0.7 Dataset loaders | Haiku 5.5 | medium | Repetitive loaders with checksum tests |
| S1.1 Claim parser | Sonnet 5.5 | high | Many linguistic edge cases |
| S1.2 Panoptic inventory | Sonnet 5.5 | medium | Standard transformers integration plus a recall table |
| S1.3 Open-vocab grounder | Sonnet 5.5 | medium | Integration; calibration is separate |
| S1.4 Three-way verdict engine | Opus 5.5 | high | Core logic; the contradiction rule decides the metric's validity |
| S1.5 Recall, components, schema | Sonnet 5.5 | medium | Mask-area math and pydantic schema |
| S1.6 Calibration | Opus 5.5 | high | Statistical correctness (split hygiene, FPR control, ROC) |
| S1.7 Harness runs and iteration | Sonnet 5.5 | high | Runs and fixes; Opus 5.5 high reviews failure analysis before any threshold or rule changes |
| S2.1 Relation geometry | Opus 5.5 | high | Fuzzy membership fitting, depth ordering, frame ambiguity |
| S2.2 Scene verifier | Sonnet 5.5 | medium | Zero-shot classifier plus calibration |
| S2.3 Attribute verifier | Sonnet 5.5 | medium | Crop and classify |
| S2.4 Count verifier and interval fitting | Sonnet 5.5 | high | Interval fitting on human counts |
| S2.5 Strictness profiles | Sonnet 5.5 | medium | Wiring policy to calibration |
| S2.6 Tests T11 to T20 | Sonnet 5.5 | high | Many datasets, per-relation reporting |
| S3.1 Caption generation, six captioners | Haiku 5.5 | medium | Script runs with revision pinning |
| S3.2 Baselines | Sonnet 5.5 | medium | Wrapping CLIPScore, PAC-S, ALOHa, pycocoevalcap; sanity-check against their papers |
| S3.3 Ablations | Sonnet 5.5 | high | Many variants scored on harness tests, bootstrap CIs |
| S3.4 OOD suite | Sonnet 5.5 | medium | Data assembly and reporting |
| S3.5 Tests T21 to T30 | Sonnet 5.5 | high | Perturbation suite, reproducibility runs |
| S4.1 Paper | Opus 5.5 | high | Drafting and claim-by-claim traceability; a final consistency pass at xhigh |
| S4.2 Release 2.0.0 | Sonnet 5.5 | medium | CLI, CI matrix, CHANGELOG, Zenodo; Haiku low for CI boilerplate and changelog formatting |
| S4.3 arXiv and workshop submission | Human | n/a | Account, endorsement and submission are the author's actions |
| Phase 5 adoption | Haiku 5.5 low for issue triage; Sonnet 5.5 medium for integration PRs | low to medium | Ongoing, low-risk work |

**Cross-cutting checks:**

- **Code review.** Run the code-review skill at high effort before every merge touching the verdict engine, calibration or harness. Use medium for everything else.
- **Security review.** Run the security-review skill before each PyPI release.
- **Lint, format, link checks.** Haiku 5.5 low.
- **Second-model check on statistics.** Any new threshold, correlation or significance claim is re-derived independently by a different Claude model (Opus checks Sonnet's work and the reverse) before it enters the paper.

---

## 6. Validation plan: claims, tests, thresholds

### 6.1 Terminology
- **CHAIR-GT**: object hallucination labels computed from COCO instance annotations with the standard CHAIR synonym list (the gold standard for object hallucination on COCO).
- **Golden set**: the 300 hand-labelled captions from S0.6.
- All numbers are reported with bootstrap 95% confidence intervals (1,000 resamples over images). Metric-level thresholds are fitted on *calib*, tuned on *dev*, reported once on *test*.

### 6.2 Gates (must pass before the paper is written)

| Gate | Statement | Threshold |
|---|---|---|
| A | VTAS-P detects object hallucinations | sentence-level AUROC ≥ 0.85 and object-level F1 ≥ 0.80 vs CHAIR-GT on test split |
| B | VTAS ranks captioning systems correctly | Spearman ≥ 0.9 between VTAS-P system ranking and CHAIR-GT system ranking over six captioners; also ≥ 0.8 vs THumB human ranking |
| C | VTAS-P tracks human precision; VTAS-R tracks human recall | Kendall τ_b vs THumB precision ≥ CLIPScore's τ; τ_b vs THumB recall ≥ CLIPScore's τ |
| D | Low false alarms on correct text | human reference captions flagged `CONTRADICTED` ≤ 5% on COCO test split, ≤ 10% on OOD sets |
| E | Spatial component is real | VSR test accuracy of VTAS-S ≥ 0.70 and above CLIPScore caption-pair accuracy; What'sUp left/right/above/below ≥ 0.75 |
| F | Policy is standardizable | annotator kappa ≥ 0.7 on verdicts; policy decisions agree with adjudicated labels ≥ 0.9 |
| G | Graceful degradation out of domain | on OOD sets, coverage drops (reported) but false-alarm rate stays within Gate D; no silent score collapse |

### 6.3 Test catalogue

Each test has an id, the claim it backs, data, metric, pass criterion, baselines compared. Tests marked **[unit]** run in CI on CPU with cached fixtures; the rest are harness runs.

**A. Code-level correctness [unit]**

| Id | Claim backed | Test | Pass |
|---|---|---|---|
| U1 | Scoring maths is correct | Golden fixtures: hand-computed P, R, F-beta, coverage for 20 synthetic claim/inventory configurations (empty caption, empty inventory, all unverifiable, duplicates, plural intervals) | exact match to 4 dp |
| U2 | Parser handles compounds | "fire hydrant", "cell phone", "hot dog", "stop sign", "teddy bear", "wine glass", "traffic light" each yield one claim | 100% |
| U3 | Parser handles counts | "two dogs", "a couple of cats", "several people", "a crowd" → correct intervals per policy | 100% |
| U4 | Parser handles attributes and relations | "a red car under a tree" → attribute(car, red), relation(car, under, tree) | 100% |
| U5 | Verdicts are three-way | A claim outside trusted vocabulary never becomes `CONTRADICTED` | property-based test (hypothesis), 0 violations |
| U6 | Instance assignment prevents collapse | 2 persons detected + "a man and a woman" → both supported; 1 person + "a man and a woman" → second claim contradicted/unverifiable per profile | 100% |
| U7 | Determinism | Same input twice → identical JSON (bit-for-bit) | 100% |
| U8 | Policy file is the source of truth | Changing an interval in YAML changes verdicts with no code change | 100% |
| U9 | Package installs clean | CI job: fresh venv, `pip install .[light]`, run CLI on a bundled sample image | exit 0 on 3.10/3.11/3.12 |
| U10 | Legacy reproduces | `vtas.legacy` on 50 archived Phase 1 images reproduces the archived scores | max abs diff ≤ 1e-3 |

**B. Hallucination detection validity**

| Id | Claim backed | Data | Metric | Pass | Baselines |
|---|---|---|---|---|---|
| T1 | VTAS-P detects object hallucinations without annotations | COCO test split, six captioners, CHAIR-GT labels | object-level P/R/F1; sentence-level AUROC, AUPRC | Gate A | CHAIR-GT (ceiling), ALOHa, CLIPScore (threshold sweep), v1.3 legacy |
| T2 | The 18.8%-type artifacts are gone | Re-classify every `CONTRADICTED` noun on test split by the M1 taxonomy (manual on a 200 sample) | share that are stuff / fragment / garbled | ≤ 10% combined (was 59%) | v1.3 legacy |
| T3 | Each verifier adds information | Leave-one-verifier-out on test split | ΔAUROC, ΔF1 with CIs | each removal lowers F1 or the verifier is dropped | — |
| T4 | Calibration beats fixed thresholds | Calibrated vs v1-style fixed cutoffs | AUROC at matched FPR | calibrated ≥ fixed at every FPR in {2,5,10}% | — |
| T5 | Detects single-noun errors | FOIL test set (one noun swapped) | foil-vs-original pairwise accuracy; foil word localisation accuracy | ≥ 0.85 pairwise | CLIPScore, ALOHa, CHAIR-GT |
| T6 | Works on long VLM captions | LLaVA-1.5 and Qwen2-VL captions (avg > 40 words) on test split | same as T1, stratified by caption length | AUROC does not drop > 0.05 vs short captions | ALOHa |

**C. Human agreement**

| Id | Claim backed | Data | Metric | Pass | Baselines |
|---|---|---|---|---|---|
| T7 | VTAS-P ≈ human precision; VTAS-R ≈ human recall | THumB 1.0 (500 COCO images, human + machine captions, precision and recall ratings) | Kendall τ_b per component | Gate C | CLIPScore, RefCLIPScore, PAC-S, CIDEr, SPICE |
| T8 | Combined score is a sane quality signal | Flickr8k-Expert, Flickr8k-CF | Kendall τ_c / τ_b | within 0.05 of CLIPScore (not expected to beat it; VTAS is a faithfulness metric) | CLIPScore, PAC-S |
| T9 | Pairwise preferences | Pascal-50S (HC, HI, HM, MM subsets) | pairwise accuracy | ≥ CHAIR-style baselines on HI (hallucination-incorrect) subset; report all four | CLIPScore, CIDEr |
| T10 | Beta is data-driven | Fit beta maximising τ with THumB overall score on calib; report on test | chosen beta with CI | reported, not hand-set | — |

**D. Policy and standardization**

| Id | Claim backed | Data | Metric | Pass |
|---|---|---|---|---|
| T11 | Policy decisions match adjudicated human judgment | Golden set | verdict accuracy; per-category (specific/general, count, synonym) | Gate F |
| T12 | Annotators agree with each other under the rubric | Golden set, two annotators | Cohen's kappa; Krippendorff's alpha | ≥ 0.7 |
| T13 | Count intervals are human-grounded | 150 images with "several / group / crowd" captions, human count estimates | fitted interval coverage of human answers | ≥ 90% coverage |
| T14 | Profiles are ordered | Perturbation suite under strict / standard / lenient | VTAS-P(strict) ≤ VTAS-P(standard) ≤ VTAS-P(lenient) per image | 100% |

**E. Spatial, scene, attributes, counts**

| Id | Claim backed | Data | Metric | Pass | Baselines |
|---|---|---|---|---|---|
| T15 | Spatial relations are verified | VSR test (zero-shot and random splits) | accuracy overall and per relation | Gate E | CLIPScore pair accuracy, random |
| T16 | Left/right/above/below are not guessed | What'sUp (controlled sets A and B) | accuracy | ≥ 0.75 | CLIP, random (0.25/0.5) |
| T17 | Depth-based relations work | VSR "in front of"/"behind" subset | accuracy with and without C6 | C6 adds ≥ 0.1 | — |
| T18 | Scene claims are verified | Golden-set scene claims + 300 Places365-val images with templated captions | accuracy | ≥ 0.85 | v1.3 CLIP gate |
| T19 | Attribute hallucinations are caught | AMBER attribute dimension (generative task) | AMBER attribute hallucination agreement; our verdicts vs AMBER annotations | F1 ≥ 0.75 | detector-score-only attribute judgement (expected to fail) |
| T20 | Counts are verified | Golden-set count claims; COCO instance counts | count-claim accuracy | ≥ 0.85 | v1.3 plural flag |

**F. Robustness, OOD, monotonicity**

| Id | Claim backed | Data | Metric | Pass |
|---|---|---|---|---|
| T21 | Scores move the right way | Synthetic perturbations of human captions on test split: add an absent object, swap an object, remove a salient object, swap a colour, flip a relation, change a count | sign of ΔVTAS-P / ΔVTAS-R / ΔVTAS-A / ΔVTAS-C / ΔVTAS-S per perturbation | ≥ 95% correct direction per perturbation type |
| T22 | Correct text is not punished | Human reference captions (five per image) on COCO test, nocaps val | `CONTRADICTED` rate per caption | Gate D |
| T23 | Out-of-domain degrades gracefully | nocaps out-domain, 200 sketches/clipart, 200 night images, 100 medical, 100 satellite (with reference captions) | false-alarm rate, coverage, share of `UNVERIFIABLE` | Gate G; coverage reported per domain |
| T24 | Detector choice is not the metric | Swap C2 between tiny/large Mask2Former and Panoptic-FPN; swap C3 between OWLv2 and Grounding DINO | ΔAUROC on T1 | ≤ 0.03 between full-profile backbones |
| T25 | Not sensitive to caption style | Same content paraphrased (3 paraphrases per 300 captions via templates) | score std within paraphrase group | ≤ 0.05 |
| T26 | Stable under image perturbation | JPEG q=50, 0.5× downscale, mild blur on 500 images | ΔVTAS-P | mean |Δ| ≤ 0.03 |

**G. System-level and reproducibility**

| Id | Claim backed | Data | Metric | Pass |
|---|---|---|---|---|
| T27 | Ranks systems like ground truth and humans | Six captioners on test split; THumB systems | Spearman vs CHAIR-GT ranking; vs THumB human ranking | Gate B |
| T28 | Ablations are honest | Every ablation row re-run end-to-end and scored on T1/T7 | table with CIs | no ablation measured by mean score |
| T29 | Numbers reproduce | Two independent runs (Kaggle T4 light; RTX PRO 6000 full) from a clean clone | all headline numbers within CI | yes |
| T30 | Cost is known | Throughput and peak VRAM per profile on T4 and RTX PRO 6000 | images / s, GB | reported in paper |

### 6.4 Statistical protocol
- Bootstrap 95% CIs over images for every number.
- Paired comparisons against baselines use paired bootstrap on the same images; report p-values for the headline comparisons (T1 vs ALOHa, T7 vs CLIPScore).
- Seeds fixed; model revisions pinned in `models.lock`.
- Calibration split is never used for reporting.

### 6.5 Paper review checklist (must all be true)
- [ ] Every number in the abstract traces to a test id above and a results JSON in the repo.
- [ ] Hyperparameters in the paper are read from `calibration.json`, not typed by hand.
- [ ] CHAIR reported is CHAIR-GT, computed from COCO instances with real image ids.
- [ ] Ablations are scored on T1/T7, never on mean VTAS.
- [ ] ALOHa, OpenCHAIR, AMBER, POPE, FaithScore, CHAIR, CLIPScore/PAC-S, THumB cited and compared where applicable.
- [ ] Limitations section names: non-commercial Mask2Former weights, VLM probe can hallucinate, left/right frame ambiguity, coverage on OOD domains.
- [ ] Policy version and strictness profile stated.

---

## 7. PyPI and compatibility plan

| Version | Content | Timing |
|---|---|---|
| 1.3.1 | Bugfix only: scipy dependency, spaCy model install, metadata (email, summary, python_requires), README dedupe, deprecation warning, no metric changes | Phase 0 |
| 1.3.x | Nothing further; v1 is frozen | — |
| 2.0.0a1 | Core v2 (Phase 1), light profile, `vtas.legacy` included, `[full]` extra marked experimental | end of Phase 1 |
| 2.0.0b1 | Spatial / scene / attribute / count components, profiles | end of Phase 2 |
| 2.0.0 | With paper; pinned model revisions; benchmark JSON; DOI | Phase 4 |

Rules: do **not** yank 1.3.0 (reproducibility of any downstream user); import path `vtas` is kept; `VTASEvaluator` remains as a thin wrapper over the legacy module with a warning; the new entry point is `vtas.Evaluator` returning a typed `Result` object with `.components`, `.claims`, `.coverage`, `.to_json()`.

---

## 8. Risks and fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| OWLv2 / Grounding DINO scores cannot be calibrated well enough for Gate A | medium | Scope 2.0 as an in-vocabulary (133-class) hallucination metric with explicit coverage; keep open-vocab as `UNVERIFIABLE` triage only |
| HAT annotations from ALOHa are not publicly downloadable | medium | Golden set (S0.6) plus CHAIR-GT replace it; request HAT from the authors in parallel |
| Mask2Former non-commercial license blocks some users | certain for commercial use | Document; ship Panoptic-FPN (Apache) as the `[oss]` backbone and report T24 for it |
| VLM probe introduces its own hallucinations | high | One vote of three; measure its standalone false-alarm rate in T22 and drop it if it hurts Gate D |
| Spatial component accuracy stays low (VSR is hard; SOTA VLMs ≈ 70%) | medium | Report VTAS-S separately and exclude it from the headline; still a contribution if above CLIP |
| Full profile too heavy for Kaggle | certain | Light profile is the default; full profile only on the RTX PRO 6000 and documented as such |
| Timeline slips (one person) | high | Gates are ordered; a paper with Gates A–D only (no spatial) is still publishable |

---

## 9. Re-verification log (2026-11-13) and changes from the earlier chat plan

Verified against sources:
- THumB 1.0 exists with public precision/recall human scores (NAACL 2022, github.com/jungokasai/THumB); 500 COCO images. Use it as the primary human-agreement set for component-wise correlation (T7).
- ALOHa code is public (`pip install git+https://github.com/DavidMChan/aloha.git`); the **HAT annotation download was not confirmed** → moved to a risk with the golden set as fallback.
- `facebook/mask2former-swin-large-coco-panoptic` loads with `Mask2FormerForUniversalSegmentation`; COCO panoptic label set is 80 things + 53 stuff; weights carry a non-commercial license → license caveat and Apache fallback added.
- `depth-anything/Depth-Anything-V2-Small-hf` runs via the `transformers` depth-estimation pipeline; Small is Apache-2.0, larger sizes are CC-BY-NC → Small chosen.
- VSR (66 relations, ~10k items on COCO images) and What'sUp (controlled left/right/above/below) are public → used for T15–T17.
- AMBER provides existence / attribute / relation annotations on 1,004 images → used for T19 and OOD-style checks.
- `merve/coco` has no image ids (confirmed on its dataset card) → official COCO files are mandatory (S0.5).
- `vtas-metric` 1.3.0 is live on PyPI with placeholder email and missing `scipy` → 1.3.1 bugfix release planned before any 2.0 work.

Changes from the earlier chat plan:
- Places365 scene classifier replaced by zero-shot SigLIP/CLIP over Places365 labels (no maintained `transformers` checkpoint).
- Added a 1.3.1 bugfix release and an explicit no-yank rule.
- Added the OOD suite (T23), paraphrase and image-perturbation stability tests (T25, T26), legacy reproduction test (U10), and the paper checklist.
- Spatial and policy work moved into their own phase with gates E and F; the headline can ship without the spatial component if it underperforms.

Sources: [THumB paper](https://arxiv.org/pdf/2111.08940) · [ALOHa paper](https://arxiv.org/html/2404.02904v1) · [Mask2Former COCO panoptic](https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic/tree/main) · [Depth Anything V2 Small hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/commit/32d03942121d29edb49de4e2cc15831558af3f36) · [VSR](https://arxiv.org/pdf/2205.00363) · [What'sUp](https://huggingface.co/papers/2310.19785) · [AMBER](https://github.com/junyangwang0410/amber) · [Bianchi et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Bianchi_The_Devil_is_in_the_Fine-Grained_Details_Evaluating_Open-Vocabulary_Object_CVPR_2024_paper.pdf)
