# VTAS Rectification: Phase Handoffs (one chat per phase)

Purpose: run each phase in a fresh chat so no chat re-reads the whole history. Every chat reads only the files and plan sections listed in its handoff, does one bounded job, and writes its result to `PROGRESS.md`.

Plan file: `VTAS_Rectification_Plan.md` (called "the plan" below; in the repo it is `docs/ROADMAP_V2.md`).
Repo: `Projects/VTAS_Evaluation_Metric/vtas-metric` (remote: github.com/Ritanshu-Prasad/VTAS_Evaluation_Metric).

---

## How to run this without wasting tokens

1. Start each phase in a new chat. Switch the model with `/model` to the one named in the handoff, then paste that handoff block as the first message.
2. Do not paste the whole plan. The handoff names the sections to read.
3. Each chat ends by appending a short entry to `PROGRESS.md` (what was done, files changed, numbers, open issues). The next chat reads only the last entry of the chats it depends on.
4. Never rerun the audit. Section 1 of the plan is settled.
5. Run chats in order unless the "Depends on" line says otherwise. Chats 3 and 4 can run in parallel. So can Chats 7 and 8 after Chat 6.
6. Every code chat works on its own branch off `v2-dev` (Chat 0 works off `main`), commits in small steps, and stops before merging. Merging is a human decision.

## Rules every chat must follow (paste at the top of each handoff)

- No metric change ships unless it improves agreement with ground truth or human judgment on the harness. Mean VTAS is never evidence.
- No hardcoded thresholds. Thresholds come from `calibration.json`.
- Gate thresholds and split files are committed before results are computed. Never adjust a threshold after seeing test-split numbers.
- Claude is never used as a judge for numbers in the paper.
- Do not commit data, weights or per-image result dumps. Commit code, split id lists, calibration and small summary JSON.
- Do not push, merge, publish to PyPI or rotate tokens unless the user says so in that chat.
- Use only Claude Opus 5.5, Sonnet 5.5 or Haiku 5.5 for the building work.
- Finish by appending to `PROGRESS.md` and listing what the next chat needs.

---

## Chat 0: Safety and the 1.3.1 hotfix (Sonnet 5.5, medium)

**Depends on:** nothing. **Branch off:** `main`.
**Read:** plan Sections 1.2 (E1 to E7), 5 (S0.1 to S0.4), 7.

> Fix the 1.3.0 engineering problems and prepare release 1.3.1 with no metric changes. Work on branch `fix/1.3.1`. Tasks: (1) repair the tests: UTF-8 `tests/__init__.py`, package imports, dict-style claim inputs, and add pure-math tests for the scoring step with a mocked pipeline; (2) packaging: add scipy, install the spaCy model automatically or through a `vtas download` command, remove the placeholder author email (ask me for the real address), correct the summary to "Visual-Truth Alignment Score", set `python_requires>=3.10`, move to `pyproject.toml` with one version source, update CITATION.cff; (3) fix `examples/batch_test_vtas.py`; (4) rewrite `docs/architecture.md` to match the real 1.3 maths and dedupe the README, adding a "preliminary results" note over the 18.8% and r=0.30 claims; (5) add GitHub Actions workflows for tests and for a trusted-publishing release (no stored token). Done when `pytest -q` is green on CPU in under 30 seconds and a fresh venv install runs the CLI. Do not push or publish. Write the result to PROGRESS.md.

## Chat 1: Ground-truth harness (Opus 5.5, high)

**Depends on:** Chat 0 (tests green). **Branch off:** `v2-dev`.
**Read:** plan Sections 2 (rules 5 and 8), 6.1, 6.4, S0.5, S0.7.

> Build the evaluation harness. Use the official COCO val2014 images, `instances_val2014.json` and `captions_val2014.json` with real image ids (not `merve/coco`). Create fixed splits by image id: calib 1,000, dev 1,000, test 3,000, with a seed, and commit only the id lists. Implement true CHAIR (CHAIRi and CHAIRs) with the standard synonym list in `harness/chair_gt.py`, plus bootstrap confidence interval and paired bootstrap helpers. Commit the gate thresholds from plan Section 6.2 in `harness/gates.yaml` before any result is computed. Add loaders with checksum tests for FOIL, THumB 1.0, Flickr8k-Expert and CF, Pascal-50S, VSR, What'sUp, AMBER and nocaps val. Flag any dataset whose download is not public. Done when CHAIR-GT on BLIP-base captions for the dev split lands in the published ballpark (CHAIRs roughly 10 to 15%) and every loader has a passing test. Write the result to PROGRESS.md.

## Chat 2: Golden set and policy rubric (Sonnet 5.5, high)

**Depends on:** Chat 1 (image id lists). **Branch off:** `v2-dev`.
**Read:** plan Section 4 and S0.6.

> Prepare the 300-caption golden set (100 BLIP, 100 LLaVA-1.5, 100 human references from the dev split) as annotation files: one row per claim with type, text, verdict (supported, contradicted, unverifiable) and a note. Write the annotation rubric from plan Section 4, including the count-interval table and the specific-versus-general rules, and a spreadsheet template for two human annotators. Put Claude pre-labels in a separate column. They must never count as an annotator. Draft `policy/concept_policy_v1.yaml` from Section 4 with the several, group, crowd and many intervals left as `fit_from_data`. Done when the files exist and a script computes Cohen's kappa and Krippendorff's alpha from two annotators' files. I will do the human labelling. Write the result to PROGRESS.md.

## Chat 3: Claim parser (Sonnet 5.5, high)

**Depends on:** Chat 2 (policy draft). **Branch off:** `v2-dev`.
**Read:** plan Sections 2, 3 (C1), 4, tests U2 to U4.

> Implement the claim parser: spaCy noun chunks with compound merging (fire hydrant, cell phone, hot dog), numerals and quantifiers mapped to count intervals read from the policy file, adjectives to attribute claims, prepositions to relation triples, and a scene-word lexicon. Output typed claim objects with pydantic. Retire the old stop-noun list as a primary mechanism. Write tests U2, U3, U4 and U8 first. Done when they pass and agreement with the golden-set claim lists is at least 95% (say so if only Claude pre-labels were available). Write the result to PROGRESS.md.

## Chat 4: Verifiers (Sonnet 5.5, medium)

**Depends on:** Chat 1. Can run alongside Chat 3. **Branch off:** `v2-dev`.
**Read:** plan Section 3 (C2, C3, C5, C9 interfaces), S1.2, S1.3, S1.5.

> Integrate the panoptic inventory (Mask2Former tiny and large), the open-vocabulary grounder (OWLv2, then Grounding DINO tiny) and, for the full profile, the Qwen2-VL-2B yes/no probe. Each verifier returns scores with boxes or masks behind a common interface. Compute the per-class recall table for the panoptic model against COCO panoptic ground truth on the calib split. Implement saliency-weighted coverage from mask area and the component output schema (VTAS-P, VTAS-R, Coverage). Pin model revisions in `models.lock`. Done when each verifier runs on the light profile on a 16 GB GPU and the recall table is saved. Write the result to PROGRESS.md.

## Chat 5: Verdict engine and calibration (Opus 5.5, high)

**Depends on:** Chats 3 and 4. **Branch off:** `v2-dev`.
**Read:** plan Sections 2, 4.4, S1.4, S1.6, tests U5, U6, T4.

> Implement the three-way verdict engine: SUPPORTED, CONTRADICTED, UNVERIFIABLE. A claim is CONTRADICTED only when at least two verifiers whose per-class recall on the calibration split is at least 0.6 agree it is absent. Out-of-vocabulary claims are UNVERIFIABLE and lower Coverage. Use instance-level Hungarian assignment so two noun phrases cannot claim the same instance. Then write the calibration module: fit every threshold at a target false-positive rate (2, 5 and 10%) on the calib split only, store `calibration.json`, and plot ROC curves. Write tests U5 and U6 as property-based tests before the code. Done when no threshold appears as a literal in the code and both tests pass. Write the result to PROGRESS.md.

## Chat 6: Core validation, gates A and D (Sonnet 5.5 high, then an Opus 5.5 high review)

**Depends on:** Chat 5. **Branch off:** `v2-dev`.
**Read:** plan Sections 6.2 (A and D), 6.3 (T1 to T6, T22), 6.4.

> Run tests T1 to T6 and T22 on the dev split, then once on the test split. Report AUROC, object-level F1 against CHAIR-GT, leave-one-verifier-out deltas and the human-caption false-alarm rate, each with bootstrap CIs, compared with ALOHa, CLIPScore and the legacy v1.3 module. Do not change any rule or threshold after seeing test numbers. If a gate fails, write a failure analysis and stop. Done when results JSONs and a short report exist. An Opus 5.5 high chat reviews the failure analysis before anything is changed. Write the result to PROGRESS.md.

## Chat 7: Spatial relations (Opus 5.5, high)

**Depends on:** Chat 6. Can run alongside Chat 8. **Branch off:** `v2-dev`.
**Read:** plan Section 3 (C6, C8), S2.1, tests T15 to T17.

> Implement relation geometry for on, under, above, below, in, left, right, near, in front and behind over mask geometry and Depth Anything V2 Small, with fuzzy membership curves fitted on the VSR training split. Accept left and right in either camera or object frame. Report VSR accuracy per relation and What'sUp accuracy, with and without depth. If overall accuracy does not beat CLIPScore pair accuracy, report VTAS-S separately and say it is excluded from the headline. Write the result to PROGRESS.md.

## Chat 8: Scene, attributes, counts, profiles (Sonnet 5.5, medium to high)

**Depends on:** Chat 6. Can run alongside Chat 7. **Branch off:** `v2-dev`.
**Read:** plan Sections 3 (C4, C7), 4.2, 4.3, S2.2 to S2.5, tests T13, T14, T18 to T20.

> Implement the scene verifier (zero-shot SigLIP over Places365 labels plus stuff-mask evidence), the attribute verifier on crops (SigLIP so400m for full, CLIP B/32 for light), and the count verifier using the policy intervals. Fit the several, group, crowd and many intervals from the human counts in the golden set and write them into the policy file. Wire the strict, standard and lenient profiles to policy and calibration FPR. Done when tests T13, T14, T18, T19 and T20 pass their thresholds, or are reported honestly if not. Write the result to PROGRESS.md.

## Chat 9: Spatial and policy validation, gates E and F (Sonnet 5.5, high)

**Depends on:** Chats 7 and 8. **Read:** plan Sections 6.2 (E and F), 6.3 (T11 to T20).

> Run tests T11 to T20 and report them against gates E and F with CIs. Include per-relation VSR results, the golden-set kappa and the count-interval coverage. Write a failure analysis for any miss. Write the result to PROGRESS.md.

## Chat 10: Captions and baselines (Sonnet 5.5 medium; caption runs may use Haiku 5.5 medium)

**Depends on:** Chat 6. Can start earlier. **Read:** plan Section 3 (captioner table and baselines), S3.1, S3.2.

> Generate captions for the 3,000-image test split and nocaps val with the six captioners (BLIP base, BLIP-2 OPT-2.7B, GIT base COCO, LLaVA-1.5-7B, Qwen2-VL-7B, InstructBLIP) with pinned revisions. Implement or wrap CLIPScore and RefCLIPScore, PAC-S, ALOHa, CHAIR-GT, and BLEU, METEOR, CIDEr and SPICE through pycocoevalcap. Sanity check each baseline against its own paper on a small subset. Done when caption files and baseline score files exist for all six captioners. Write the result to PROGRESS.md.

## Chat 11: Ablations, OOD and robustness, gates B, C, G (Sonnet 5.5, high)

**Depends on:** Chats 9 and 10. **Read:** plan Sections 6.2 (B, C, G), 6.3 (T7 to T10, T21 to T30), S3.3 to S3.5.

> Run the ablations on harness tests (never on mean VTAS), the human-agreement tests on THumB, Flickr8k and Pascal-50S, the perturbation and OOD suites, backbone swaps, paraphrase and image-perturbation stability, ranking correlation across six captioners, and the two-machine reproducibility run (light profile on a T4, full profile on the RTX PRO 6000). Fit beta on the calib split from THumB. Report everything with CIs. Write the result to PROGRESS.md.

## Chat 12: Paper (Opus 5.5, high; final pass at xhigh)

**Depends on:** Chat 11, with at least gates A to D passed. **Read:** plan Sections 2, 4, 6.5, 8, and PROGRESS.md.

> Draft the paper as an annotation-free, claim-level hallucination metric with coverage-aware verdicts. Every number must trace to a test id and a results file, and hyperparameters are read from `calibration.json`. Cover related work: CHAIR, ALOHa, OpenCHAIR, AMBER, POPE, FaithScore, CLIPScore, PAC-S, THumB. State the policy version, the profile and the limitations (non-commercial Mask2Former weights, the VLM probe can hallucinate, left and right frame ambiguity, OOD coverage). Tick every item in plan Section 6.5. If any gate failed, write the paper at the scope the passing gates support.

## Chat 13: Release 2.0 (Sonnet 5.5, medium; Haiku 5.5 low for boilerplate)

**Depends on:** Chat 12. **Read:** plan Section 7.

> Prepare 2.0.0: extras `[light]`, `[full]` and `[legacy]`, a CLI (`vtas score`, `vtas eval`, `vtas calibrate`), GPU and batching support, caching, a CI matrix on Python 3.10 to 3.12, CHANGELOG, CONTRIBUTING, pinned model revisions, Zenodo metadata and the benchmark JSON. Run a security review before tagging. Do not yank 1.3.0. Do not publish; I will tag and publish.

---

## PROGRESS.md entry template

```
## Chat N: <name>   (date, model, effort)
Branch: <name>   Commits: <short hashes>
Done: <3 to 6 bullets>
Numbers: <test id: value [CI]>
Gate status: <A..G pass / fail / not run>
Open issues: <bullets>
Next chat needs: <files, decisions>
```
