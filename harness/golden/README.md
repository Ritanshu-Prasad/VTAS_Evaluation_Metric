# Golden Set Annotation Rubric (v1, draft)

This rubric is for the people who label the VTAS golden set. It turns
Concept Policy v1 (`policy/concept_policy_v1.yaml`) into rules a human
can apply to one image and one caption. Read it once in full before labelling, and keep
it open while you work.

The golden set has two uses:

1. **Inter-annotator agreement.** Gate F needs Cohen's kappa ≥ 0.7 on verdicts. The
   threshold is in `harness/gates.yaml`.
2. **Ground truth for VTAS 2.0.** This covers claim-level verdicts and the count
   judgements that fix the "several / group / crowd / many" intervals.

Labels come from two humans, who work independently. Model pre-labels sit in a separate
column of the adjudication file. Annotators never see them, and they never count as an
annotator.

---

## 1. What you label

Each row is one **claim**: a single checkable statement that the caption makes about the
image. The claims are split from the caption before labelling starts and are then frozen.
For each row you fill in:

| Column | When | Values |
|---|---|---|
| `verdict` | every row | `supported`, `contradicted`, `unverifiable` |
| `visible_count` | every `count` row; `object` rows only in the case in 4.3 | integer ≥ 0 |
| `flag` | only if the claim unit itself is wrong | `not_a_claim`, `wrong_type`, `should_split`, `should_merge`, `unclear_text` |
| `note` | required for `contradicted` and `unverifiable`; optional otherwise | free text: what you saw |

Rules for the file:
- Do not edit `claim_id`, `type` or `claim_text`.
- Do not sort and save, and do not insert or delete rows.
- If the caption makes a claim that has no row, add it to the `missing_claims` sheet.
  Missing claims are adjudicated, but they do not enter the agreement statistics.
- Even when you flag a row, still give a verdict for the claim as written.

**Look only at the image.** Do not use the other captions of the same image, web
searches or your memory of COCO. Do not discuss labels with the other annotator until
both files are finished.

The files are **blinded**. Captions carry opaque ids (`g001`…`g350`), and the source
(BLIP, LLaVA or human) is not shown. Some images appear up to four times, with different
captions. Judge each caption on its own.

---

## 2. Claim types

Every claim has one of five types. The `claim_text` follows a fixed shape for each type.

| Type | What it asserts | `claim_text` shape | Example from "two brown dogs sleeping on a couch in a living room" |
|---|---|---|---|
| `object` | at least one instance of a thing, person or animal is visible | noun phrase without article or number | `dogs`, `couch` |
| `count` | how many instances there are | the caption's own quantity expression | `two dogs` |
| `attribute` | a property of one entity: colour, size, material, age, state, pose, single-entity action | `<entity>: <attribute>` | `dogs: brown`, `dogs: sleeping` |
| `relation` | a link between two entities: spatial, or an action with an object | `<subject> / <predicate> / <object>` | `dogs / on / couch` |
| `scene` | the setting or place type, or global conditions | the scene term | `living room` |

### 2.1 Splitting rules (used to make the rows; use them to judge `flag` and `missing_claims`)

- **One noun phrase gives one `object` claim.** Compound nouns stay whole: `fire hydrant`,
  `teddy bear`, `glass of beer`.
- **`count` claims are added only for explicit quantity:**
  - numerals ("two", "three")
  - bare plurals of countable things ("trees", "plates")
  - quantifiers ("some", "a few", "several", "many", "lots of", "a bunch of", "multiple")
  - collective nouns ("a group of", "a crowd", "a herd of", "a couple of", "a pair of")

  A plural phrase gets both an `object` row ("are there any?") and a `count` row ("are
  there that many?").
- **No `count` claim** for:
  - singular "a / the / one X"; see 4.3 for how those are checked
  - mass nouns and food mixtures: water, grass, snow, beans, fries, teeth
  - nouns that come as one item: "a pair of scissors", "a pair of skis", "glasses"
- **Each adjective, colour, material or state gives one `attribute` claim.** So "a small
  open window" gives `window: small` and `window: open`.
- **Actions:**
  - An action with an object is a `relation`: `man / holding / umbrella`.
  - An action without one is an `attribute`: `man: walking`.
- **Prepositions:**
  - A preposition between two entities is a `relation`: `cup / on / table`.
  - "X in/on/at <place>" gives a `scene` claim for the place, with no extra relation.
    Example: "a dog on the beach" gives `dog`, `beach`.
  - A positional phrase relative to the place is kept as a `relation`: "in the center
    of a park", "on the side of the road".
- **Scene terms:**
  - room types: kitchen, bathroom, bedroom, living room
  - outdoor places: beach, street, road, field, park, forest, ocean, market, station,
    zoo, city
  - events: baseball game
  - global conditions: at night, indoor, snow (when it describes the setting)
- **Subjective or non-visual content still gets a row:** "delicious", "for sale", "of
  great interest to them". These are usually `unverifiable` (see 3).

The draft segmentation was made automatically from the caption **text only**, before anyone
looked at the images. It contains no verdicts. The owner reviews it before labelling
starts.

---

## 3. Verdicts

Decide **only from what is visible in this image**:

- **`supported`**: you can see that the claim is true. Small, blurred or partly hidden
  instances count when you are confident what they are.
- **`contradicted`**: you can see that the claim is false.
  - The object is absent, even though the area where it would be is visible.
  - The count is outside the interval in Section 4.
  - The attribute is clearly different (a brown couch called "red").
  - The relation is clearly different (cup under the table, not on it).
  - The scene is clearly a different place type.
- **`unverifiable`**: the image cannot settle it either way.
  - It is too small, cropped, dark or blurred to tell.
  - It is not visual: taste, intention, sound, "for sale" without a sign.
  - It depends on time: "about to throw", "landing", "pulling up".
  - Several readings are equally good.

Decision order when unsure:

1. Could a careful viewer point at evidence **for** the claim? If yes, label `supported`.
2. Could they point at evidence **against** it? If yes, label `contradicted`.
3. Otherwise label `unverifiable`.

Do not use `unverifiable` as a polite `contradicted`. If the thing is clearly not
there, the claim is `contradicted`.

**Garbled words.** A caption may contain broken words ("a gi in a zoo", "playing with a
frc") or repeated words ("pew pew pew…").
- If the intended word is not obvious, mark the row `flag = unclear_text`, label it
  `unverifiable`, and write your best reading in `note`.
- Do not silently repair the caption.

**Depictions.** Things shown in a picture, poster, screen, reflection or statue count as
depictions. Examples: a dog on a TV screen; "stuffed bears" for teddy bears.
- If the caption refers to the depiction itself (for example "a poster of a dog"), it is
  `supported`.
- If the caption implies the real thing (for example "a dog" when only a TV dog is
  visible), it is `contradicted`, with a note.

---

## 4. Counts

### 4.1 Interval table

Count claims are checked against an interval of visible instances. Label verdicts with
the **standard** profile:

| Expression | Standard interval | Strict | Lenient |
|---|---|---|---|
| "a X", "one X", "the X" | [1, 2] | [1, 1] | [1, ∞) |
| "two X", "three X", … (N) | [N, N] | [N, N] | [N−1, N+1] |
| bare plural "Xs", "some X", "a few X" | [2, ∞) | [2, ∞) | [2, ∞) |
| "a couple of X", "a pair of X" | [2, 2] | [2, 2] | [2, 3] |
| "several X" | **fit from data** (prior [3, ∞)) | | |
| "a group of X" (also "a bunch of", "a herd of", "multiple", "lots of") | **fit from data** (prior [3, ∞)) | | |
| "a crowd", "crowds" | **fit from data** (prior [6, ∞)) | | |
| "many X", "a lot of X" | **fit from data** (prior [6, ∞)) | | |

The strict and lenient columns are listed for reference only. Strict and lenient verdicts
are worked out later from your `visible_count`, so you label only standard.

### 4.2 Vague quantifiers ("several", "group", "crowd", "many" and similar)

The policy does not fix these intervals yet; your labels are the data that will fix
them. For these rows:

1. **Count.** Enter `visible_count`: the number of instances of that noun you can see.
   - Count partly visible instances.
   - Do not count instances you only assume are outside the frame.
   - If there are more than 20, give your best estimate and write "estimate" in the
     note.
2. **Verdict.** Use your own judgement as a fluent English speaker: would you call this
   many "a group", "several", "a crowd" or "many"?
   - Natural for the number you see: `supported`.
   - Clearly wrong (for example "a crowd" for two people): `contradicted`.
   - Borderline: `unverifiable`, with a note.

### 4.3 Singular phrases

There is no count row for "a X", "the X" or "one X". The `object` row asks only whether
at least one X is visible.

If the caption says "a X" but **three or more** X are clearly visible, still label the
object row `supported`. Then enter the number in `visible_count` on that `object` row, so
the strict profile ([1, 1]) and the standard profile ([1, 2]) can be checked later.

### 4.4 Counting across phrases

Each noun phrase claims its own instances, and the same instance cannot count for two
phrases. Example: for "a man and two people" you need three people in total.

---

## 5. Specific versus general terms

- **General terms.** A general term (person, animal, vehicle, food, dish, plant) is
  `supported` by any visible instance of a more specific kind. "A person" is supported by
  a visible boy, woman, player or rider.
- **Specific terms.** For a specific term (boy, woman, policeman, seagull, steer, SUV,
  cupcake), check the parent class first, then the distinguishing feature:

  | What you see | Verdict |
  |---|---|
  | Parent visible, feature clearly right | `supported` |
  | Parent visible, feature clearly wrong (an adult man called "boy"; a pigeon called "seagull") | `contradicted` |
  | Parent visible, feature cannot be told (gender of a distant figure; species of a small bird) | `unverifiable` |
  | Parent not visible | `contradicted` |

- **Synonyms and near-synonyms** are not errors. Examples: couch/sofa, bike/bicycle,
  plane/airplane, cupboard/cabinet, rug/carpet.

**This differs from the automatic verifier.** In the standard profile, the automatic
verifier never turns a failed soft attribute check into CONTRADICTED. That rule is about
the *automatic* checker's uncertainty. You can see the image, so label what is true: a
specific term that is clearly wrong is `contradicted`. Gate F compares the policy with
your adjudicated labels, so mismatches of this kind are what the gate is meant to catch.

---

## 6. Other attributes and scenes

- **Colours.** Judge the main colour under normal lighting. Mixed or ambiguous colours
  (white vs light grey, navy vs black) are `unverifiable`, not `contradicted`.
- **Size words** (small, large, big, little, tall) are relative to what is normal for
  that kind of thing. Only contradict obvious mismatches.
- **Age words** (young, little, baby, adult, old).
  - `supported` when visually clear.
  - `contradicted` only when clearly wrong.
  - `unverifiable` otherwise.
- **Single-frame actions.**
  - Poses you can see (sitting, standing, lying, jumping in the air) can be supported or
    contradicted.
  - Intent and future actions ("about to", "gets ready to", "headed in for landing") are
    `unverifiable`, unless the pose clearly contradicts them.
- **Scenes.** The place type has to fit what is visible.
  - A kitchen corner seen in a living-room photo still supports "kitchen" only if kitchen
    fixtures are visible.
  - "Beach" needs sand and/or sea shore.
  - "Field" fits any open grassy or dirt area.
- **The photo itself.** "An old picture", "black and white photo" and "blurry" are
  attributes of the image: label them like any other attribute.

---

## 7. Process

1. **Segmentation review (owner).** Read the claim rows in the adjudication preview. Fix
   the frozen claim files under `harness/golden/claims_src/` where the splitting rules in
   2.1 were broken. After that, the rows are frozen.
2. **Build the files.**

   ```
   python -m harness.golden.annotate build
   ```

   This refuses to run until every caption has text and rows. It writes
   `data/golden/annotator_A.xlsx`, `annotator_B.xlsx` and `adjudication.xlsx`. These are
   not committed.
3. **Independent labelling.** Each annotator fills in their own copy.
   - Neither looks at the other's file, at the adjudication file or at the model
     pre-labels.
   - Label all 350 captions. Core agreement is computed on blocks P and U.
4. **Agreement on the 300-caption core.**

   ```
   python -m harness.golden.agreement data/golden/annotator_A.xlsx data/golden/annotator_B.xlsx --blocks P,U --out harness/results/golden_agreement_core.json
   ```

   This reports Cohen's kappa with a bootstrap 95% CI over images, Krippendorff's alpha,
   per-source and per-type breakdowns, and the gate F check. Run it again with
   `--blocks L` for the long-caption extension; that run is reported separately and is
   not part of the gate.
5. **Merge and adjudicate.**

   ```
   python -m harness.golden.annotate merge data/golden/annotator_A.xlsx data/golden/annotator_B.xlsx
   ```

   This copies both labels into `adjudication.xlsx`. The adjudicator fills in
   `final_verdict` and `final_visible_count`, plus `adjudication_note` when the two
   annotators disagreed.
   - Model pre-labels may be shown to the adjudicator only after both annotations are
     final.
   - Their agreement with the final labels is reported separately.
6. **Fit the vague-quantifier intervals.** Use the adjudicated `visible_count` on
   "several", "group", "crowd" and "many" rows. Freeze the result in
   `policy/concept_policy_v1.yaml` (see the `fit:` block there) before any VTAS 2.0
   result is computed on the test split.

## 8. Golden-set composition

Composition: 200 dev images, seed 0. The full design is in `harness/golden/design.py`.

| Block | Images | Captions | Use |
|---|---|---|---|
| P (paired) | 50 | BLIP + LLaVA-1.5 short + 1 human reference each = 150 | compare sources on the same images |
| U (unpaired) | 150 | 50 BLIP + 50 LLaVA short + 50 human = 150 | breadth; count-interval fitting |
| L (extension) | the 50 P images | LLaVA-1.5 long ("describe in detail") = 50 | short- vs long-caption behaviour; not in the gate |

Caption sources:
- **BLIP (ViT-B):** original Salesforce code and COCO caption checkpoint, paper decoding
  (`harness/caption_blip_orig.py`).
- **LLaVA-1.5-7B:** `harness/kaggle/llava15_golden_captions.py`, pinned revision,
  greedy decoding.
- **Human:** one official COCO reference per image, drawn with the seed.
