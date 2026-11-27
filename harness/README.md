# VTAS evaluation harness

Ground truth, fixed splits, statistics, pre-registered gates and external dataset
loaders for VTAS 2.0. Not shipped in the
`vtas-metric` wheel. Run everything from the repo root. Install with `pip install -e ".[harness]"`.

Data goes to `$VTAS_DATA` (default `./data/`, gitignored). Never commit data, captions,
weights or per-image dumps. Commit code, split id lists and small summary JSON only.

| File | What |
|---|---|
| `gates.yaml` | Gate A–G thresholds, committed before any result. Do not edit after seeing test numbers. |
| `splits/coco_{calib,dev,test}.txt` | 1,000 / 1,000 / 3,000 COCO val2014 image ids (Karpathy test split, seed 0). `coco_manifest.json` has the hashes. |
| `chair_gt.py` | CHAIR-GT (CHAIRs, CHAIRi), a faithful port of Rohrbach et al. 2018. |
| `stats.py` | `bootstrap_ci`, `paired_bootstrap` (1,000 resamples over images, seed 0). |
| `coco.py` | Official COCO annotations and per-image downloads with checksums; split tools. |
| `caption_blip_orig.py` | BLIP captions with the original Salesforce code and COCO caption checkpoint (pinned commit, SHA-256 checked, weights-only load). Use this one. |
| `caption_blip.py` | **Defective, do not use**: the Hugging Face BLIP-base checkpoint drops WordPiece continuations ("gi" for giraffe). Kept for the record. |
| `eval_chair.py` | CHAIR-GT summary for a captions file on a split. |
| `golden/` | Golden set: `design.py` + `golden_ids.json` (which captions), `claims_src/` (frozen claim rows; local only, gitignored because they quote COCO and model captions), `annotate.py` (annotator and adjudication workbooks), `agreement.py` (Cohen's kappa, Krippendorff's alpha), `llava_check.py`. Rubric: `harness/golden/README.md`. |
| `kaggle/` | Self-contained scripts for runs that do not fit the local GPU (LLaVA-1.5 golden-set captions). |
| `datasets/` | Registry and loaders: FOIL, THumB 1.0, Flickr8k-Expert/CF, Pascal-50S, VSR, What'sUp, AMBER, nocaps val. |

```bash
python -m harness.coco annotations            # official COCO 2014 annotations (checksummed)
python -m harness.coco images --split dev     # official val2014 images for a split
python -m harness.caption_blip_orig --split dev   # original BLIP -> data/captions/ (needs 4.27 <= transformers < 4.50)
python -m harness.eval_chair data/captions/blip_orig_paper_dev.json --split dev --out harness/results/chair_gt_blip_orig_paper_dev.json --check-tokenizer
python -m harness.golden.annotate build       # golden-set workbooks -> data/golden/
python -m harness.golden.agreement A.xlsx B.xlsx --blocks P,U   # kappa / alpha
python -m harness.datasets                    # status of the external sets
python -m harness.datasets fetch              # download all fetchable sets
```

Protocol: fit thresholds on **calib**, tune on **dev**, report once on
**test**. Never report calib numbers.

Access flags: Pascal-50S has no working download, Flickr8k needs a sign-up form at the
original source (a mirror is used), and the official nocaps val link returns 403 (a mirror is used).
See `harness/datasets/__init__.py`.

Overlap: all 500 THumB images are in the splits (104 calib, 95 dev, 301 test). FOIL
test covers 793 calib, 823 dev and 2,400 test images. Exclude calib images when reporting on these sets.

## Claim parser agreement

The VTAS 2.0 claim parser (`src/vtas/claims/`) turns a caption into typed claims: `object`, `count`, `attribute`, `relation` and `scene`. Count intervals are read from `policy/concept_policy_v1.yaml`, never from code.

```python
from vtas.claims import ClaimParser, ConceptPolicy
parser = ClaimParser(model="sm")            # "trf" (en_core_web_trf) is the default
r = parser.parse("two brown dogs on a couch")
for c in r.claims:
    print(c.type, c.text)                   # object dogs / count two dogs / attribute dogs: brown / ...
r.of_type("count")[0].interval(ConceptPolicy.load(), "lenient")
```

`several`, `group`, `crowd` and `many` return a provisional prior interval until the fitted values are frozen in the policy file.

**Metric.** Fixed before the parser was first run (`harness/golden/claim_agreement.py`):

- A claim is `(type, normalised text)`: lower-case, articles removed, punctuation other than the ` / ` and `: ` separators removed.
- Parser and reference claims are matched per caption as multisets and summed over captions; agreement is F1 = 2TP / (P + R).
- Target: pooled agreement of at least 95% over the 300-caption core (BLIP, LLaVA short, human), with a bootstrap 95% CI over images. LLaVA long captions are reported separately.
- Split (`harness/golden/claim_parser_split.json`): 175 dev and 175 held-out captions, by image id, stratified by block and source, seed 0. Rules are developed on dev only; held-out is evaluated once.

```
python -m harness.golden.claim_agreement --model {sm,trf} --part {dev,heldout,all}
```

**Current result** (reference lists after convention normalisation, `harness/results/claim_parser_*_v2.json`):

| spaCy model | Dev | Held-out |
|---|---|---|
| `en_core_web_trf` | 85.6% | 77.9% |
| `en_core_web_sm` | 82.6% | 74.6% |

The 95% target is not met yet. Most of the remaining disagreement comes from the reference lists rather than the parser: the same construction is segmented differently in different captions, and some rows need world knowledge or coreference ("skateboarding" as `boy / riding / skateboard`, "he" as `man`). The remaining parser errors are mainly prepositional-phrase attachment, nested "of" phrases and coordination. The result files without the `_v2` suffix are the measurements taken before normalisation.

**Limits.** The reference lists were drafted automatically and reviewed by the author; they are not independent human segmentations. Coreference ("he", "it") is not resolved, and there is no LLM-based extraction mode yet.
