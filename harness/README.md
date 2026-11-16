# VTAS evaluation harness

Ground truth, fixed splits, statistics, pre-registered gates and external dataset
loaders for VTAS 2.0 (plan `docs/ROADMAP_V2.md`, S0.5 and S0.7). Not shipped in the
`vtas-metric` wheel. Run everything from the repo root. Install with `pip install -e ".[harness]"`.

Data goes to `$VTAS_DATA` (default `./data/`, gitignored). Never commit data, captions,
weights or per-image dumps. Commit code, split id lists and small summary JSON only.

| File | What |
|---|---|
| `gates.yaml` | Gate A–G thresholds from plan 6.2, committed before any result. Do not edit after seeing test numbers. |
| `splits/coco_{calib,dev,test}.txt` | 1,000 / 1,000 / 3,000 COCO val2014 image ids (Karpathy test split, seed 0). `coco_manifest.json` has the hashes. |
| `chair_gt.py` | CHAIR-GT (CHAIRs, CHAIRi), a faithful port of Rohrbach et al. 2018. |
| `stats.py` | `bootstrap_ci`, `paired_bootstrap` (1,000 resamples over images, seed 0). |
| `coco.py` | Official COCO annotations and per-image downloads with checksums; split tools. |
| `caption_blip.py` | BLIP-base captions at a pinned revision. |
| `eval_chair.py` | CHAIR-GT summary for a captions file on a split. |
| `golden/` | Golden set (S0.6): `design.py` + `golden_ids.json` (which captions), `claims_src/` (frozen claim rows), `annotate.py` (annotator and adjudication workbooks), `agreement.py` (Cohen's kappa, Krippendorff's alpha), `llava_check.py`. Rubric: `docs/GOLDEN_SET_RUBRIC.md`. |
| `kaggle/` | Self-contained scripts for runs that do not fit the local GPU (LLaVA-1.5 golden-set captions). |
| `datasets/` | Registry and loaders: FOIL, THumB 1.0, Flickr8k-Expert/CF, Pascal-50S, VSR, What'sUp, AMBER, nocaps val. |

```bash
python -m harness.coco annotations            # official COCO 2014 annotations (checksummed)
python -m harness.coco images --split dev     # official val2014 images for a split
python -m harness.caption_blip --split dev --variant paper   # -> data/captions/
python -m harness.eval_chair data/captions/blip_base_paper_dev.json --split dev \
       --out harness/results/chair_gt_blip_base_paper_dev.json --check-tokenizer
python -m harness.golden.annotate build       # golden-set workbooks -> data/golden/
python -m harness.golden.agreement A.xlsx B.xlsx --blocks P,U   # kappa / alpha
python -m harness.datasets                    # status of the external sets
python -m harness.datasets fetch              # download all fetchable sets
```

Protocol (plan 6.1/6.4): fit thresholds on **calib**, tune on **dev**, report once on
**test**. Never report calib numbers.

Access flags: Pascal-50S has no working download, Flickr8k needs a sign-up form at the
original source (a mirror is used), and the official nocaps val link returns 403 (a mirror is used).
See `harness/datasets/__init__.py`.

Overlap: all 500 THumB images are in the splits (104 calib, 95 dev, 301 test). FOIL
test covers 793 calib, 823 dev and 2,400 test images. Exclude calib images when reporting on these sets.
