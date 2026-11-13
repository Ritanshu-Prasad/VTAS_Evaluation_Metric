"""CHAIR-GT: ground-truth object hallucination on COCO (CHAIRi and CHAIRs).

A faithful Python 3 port of the reference implementation by Rohrbach et al.,
"Object Hallucination in Image Captioning" (EMNLP 2018),
https://github.com/LisaAnne/Hallucination/blob/6e4d33c4bedf6dfcd24c61bf527f40714717be47/utils/chair.py

What is kept exactly as in the reference (so numbers are comparable to published CHAIR):
  * the synonym list (``resources/chair_synonyms.txt``, sha256 below), parsed with
    ``line.strip().split(', ')``; the first entry of each line is the COCO object;
  * lower-casing, NLTK Treebank word tokenisation, Pattern 3.6 ``singularize``;
  * the double-word table, the baby/adult-animal and passenger-vehicle rules,
    ``bow tie -> tie``, ``toilet seat -> toilet``, ``wine glas -> wine glass``, and the
    "drop 'seat' when 'toilet' is present" rule;
  * ground-truth objects per image = COCO instance categories UNION the COCO objects
    mentioned in the image's human reference captions;
  * CHAIRs = hallucinated captions / captions;
    CHAIRi = hallucinated object mentions / object mentions (repeated mentions count).

Known quirks of the reference that are deliberately reproduced, not fixed:
  * Pattern singularises "bus" to "bu", "glass" to "glas" and "dress" to "dres",
    so a singular "bus" is never counted (plural "buses" is).
  * Synonym entries with a stray double space in the file (" motor bike", " cheesecake")
    can never match; "iPhone" never matches after lower-casing.

The only deviation: the reference calls ``nltk.word_tokenize``, which needs the Punkt
model download. Here sentences are split with a regular expression and each sentence is
tokenised with the same ``NLTKWordTokenizer`` that ``word_tokenize`` uses, so no data
download is needed. ``scripts`` in the harness check that this gives the same COCO
object mentions as ``nltk.word_tokenize`` on the evaluated captions.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from harness._pattern_singularize import singularize
from harness.stats import CI, bootstrap_ci, ratio_of_sums

RESOURCES = Path(__file__).resolve().parent / "resources"
SYNONYMS_PATH = RESOURCES / "chair_synonyms.txt"
SYNONYMS_SHA256 = "b685537b36fe21b791c87151b96a991ceb582edff19f3da204dcc7a7905502f2"

# Reference rules, copied from utils/chair.py (CHAIR.__init__).
COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign", "parking meter",
    "suit case", "sports ball", "baseball bat", "baseball glove", "tennis racket",
    "wine glass", "hot dog", "cell phone", "mobile phone", "teddy bear", "hair drier",
    "potted plant", "bow tie", "laptop computer", "stove top oven", "hot dog",
    "teddy bear", "home plate", "train track",
]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                "zebra", "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]


def _build_double_word_dict() -> dict[str, str]:
    d = {w: w for w in COCO_DOUBLE_WORDS}
    for a in ANIMAL_WORDS:
        d[f"baby {a}"] = a
        d[f"adult {a}"] = a
    for v in VEHICLE_WORDS:
        d[f"passenger {v}"] = v
    d["bow tie"] = "tie"
    d["toilet seat"] = "toilet"
    d["wine glas"] = "wine glass"
    return d


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_word_tokenizer = None


def word_tokenize(text: str) -> list[str]:
    """``nltk.word_tokenize`` without the Punkt download (see module docstring)."""
    global _word_tokenizer
    if _word_tokenizer is None:
        from nltk.tokenize import NLTKWordTokenizer

        _word_tokenizer = NLTKWordTokenizer()
    return [tok for sent in _SENT_SPLIT.split(text) if sent
            for tok in _word_tokenizer.tokenize(sent)]


@dataclass
class CaptionResult:
    image_id: int
    caption: str
    words: list[str]                 # COCO words as they appear (after singularize)
    objects: list[str]               # canonical COCO object per mention
    hallucinated: list[tuple[str, str]]  # (word, object) pairs not in the ground truth
    gt_objects: list[str]

    @property
    def is_hallucinated(self) -> bool:
        return bool(self.hallucinated)


@dataclass
class ChairResult:
    chair_s: float
    chair_i: float
    n_captions: int
    n_images: int
    n_mentions: int
    n_hallucinated_mentions: int
    captions: list[CaptionResult] = field(repr=False)

    def per_image_counts(self) -> np.ndarray:
        """Rows: images. Columns: hallucinated captions, captions,
        hallucinated mentions, mentions. Used for bootstrapping over images."""
        acc: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for c in self.captions:
            row = acc[c.image_id]
            row[0] += int(c.is_hallucinated)
            row[1] += 1
            row[2] += len(c.hallucinated)
            row[3] += len(c.objects)
        return np.array([acc[k] for k in sorted(acc)], dtype=float).reshape(-1, 4)

    def confidence_intervals(self, n_resamples: int = 1000, ci: float = 0.95,
                             seed: int = 0) -> dict[str, CI]:
        """Bootstrap CIs over images (plan Section 6.4)."""
        counts = self.per_image_counts()
        return {
            "CHAIRs": bootstrap_ci(counts[:, [0, 1]], ratio_of_sums, n_resamples, ci, seed),
            "CHAIRi": bootstrap_ci(counts[:, [2, 3]], ratio_of_sums, n_resamples, ci, seed),
        }

    def summary(self, n_resamples: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
        cis = self.confidence_intervals(n_resamples, ci, seed)
        return {
            "CHAIRs": cis["CHAIRs"].as_dict(),
            "CHAIRi": cis["CHAIRi"].as_dict(),
            "n_captions": self.n_captions,
            "n_images": self.n_images,
            "n_object_mentions": self.n_mentions,
            "n_hallucinated_mentions": self.n_hallucinated_mentions,
            "bootstrap": {"resamples": n_resamples, "ci": ci, "seed": seed, "unit": "image"},
        }


class ChairGT:
    """CHAIR evaluator. Build with :meth:`from_coco`, then call :meth:`evaluate`."""

    def __init__(self, synonyms_path: str | Path = SYNONYMS_PATH):
        lines = Path(synonyms_path).read_text(encoding="utf-8").splitlines()
        synonyms = [s.strip().split(", ") for s in lines]
        self.mscoco_objects: set[str] = set()
        self.inverse_synonym_dict: dict[str, str] = {}
        for syn in synonyms:
            self.mscoco_objects.update(syn)
            for s in syn:
                self.inverse_synonym_dict[s] = syn[0]
        self.double_word_dict = _build_double_word_dict()
        self.imid_to_objects: dict[int, set[str]] = {}

    # ---- caption parsing ---------------------------------------------------------------

    def caption_to_words(self, caption: str):
        """Return (words, node_words, idxs, double_words) exactly as the reference."""
        words = [singularize(w) for w in word_tokenize(caption.lower())]

        i = 0
        double_words, idxs = [], []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i:i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # "the seat of the toilet" must not fire for chair.
        if "toilet" in words and "seat" in words:
            words = [w for w in words if w != "seat"]

        # NB: the reference filters idxs against the *filtered* word list; kept as is.
        idxs = [idxs[k] for k, w in enumerate(words) if w in self.mscoco_objects]
        words = [w for w in words if w in self.mscoco_objects]
        node_words = [self.inverse_synonym_dict[w] for w in words]
        return words, node_words, idxs, double_words

    # ---- ground truth -------------------------------------------------------------------

    @classmethod
    def from_coco(cls, instances_json: str | Path | Mapping, captions_json: str | Path | Mapping,
                  image_ids: Iterable[int] | None = None,
                  synonyms_path: str | Path = SYNONYMS_PATH) -> "ChairGT":
        """Ground truth from official COCO ``instances_*.json`` and ``captions_*.json``.

        ``image_ids`` limits the ground truth to those images (all images in the
        captions file when None). Arguments may be paths or already-loaded dicts.
        """
        self = cls(synonyms_path)
        inst = _load_json(instances_json)
        caps = _load_json(captions_json)
        if image_ids is None:
            image_ids = [im["id"] for im in caps["images"]]
        self.imid_to_objects = {int(i): set() for i in image_ids}

        id_to_name = {c["id"]: c["name"] for c in inst["categories"]}
        for ann in inst["annotations"]:
            objs = self.imid_to_objects.get(ann["image_id"])
            if objs is not None:
                objs.add(self.inverse_synonym_dict[id_to_name[ann["category_id"]]])
        for ann in caps["annotations"]:
            objs = self.imid_to_objects.get(ann["image_id"])
            if objs is not None:
                objs.update(self.caption_to_words(ann["caption"])[1])
        return self

    def set_ground_truth(self, imid_to_objects: Mapping[int, Iterable[str]]) -> "ChairGT":
        """Set ground-truth objects directly (canonical COCO names); mainly for tests."""
        self.imid_to_objects = {int(k): set(v) for k, v in imid_to_objects.items()}
        return self

    # ---- scoring ------------------------------------------------------------------------

    def score_caption(self, image_id: int, caption: str) -> CaptionResult:
        if image_id not in self.imid_to_objects:
            raise KeyError(f"no ground truth for image id {image_id}")
        gt = self.imid_to_objects[image_id]
        words, node_words, _, _ = self.caption_to_words(caption)
        halluc = [(w, n) for w, n in zip(words, node_words) if n not in gt]
        return CaptionResult(image_id, caption, words, node_words, halluc, sorted(gt))

    def evaluate(self, captions: Mapping[int, str] | Iterable[tuple[int, str]]) -> ChairResult:
        """Score generated captions. Accepts {image_id: caption} or (image_id, caption) pairs
        (several captions per image are allowed; bootstrap still resamples images)."""
        items = captions.items() if isinstance(captions, Mapping) else captions
        results = [self.score_caption(int(i), c) for i, c in items]
        if not results:
            raise ValueError("no captions to evaluate")
        n_mentions = sum(len(r.objects) for r in results)
        n_halluc = sum(len(r.hallucinated) for r in results)
        return ChairResult(
            chair_s=sum(r.is_hallucinated for r in results) / len(results),
            chair_i=n_halluc / n_mentions if n_mentions else 0.0,
            n_captions=len(results),
            n_images=len({r.image_id for r in results}),
            n_mentions=n_mentions,
            n_hallucinated_mentions=n_halluc,
            captions=results,
        )


def _load_json(obj):
    if isinstance(obj, Mapping):
        return obj
    with open(obj, encoding="utf-8") as f:
        return json.load(f)
