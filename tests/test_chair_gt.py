"""CHAIR-GT unit tests: no COCO download needed (synthetic annotations)."""

import hashlib

import pytest

from harness._pattern_singularize import singularize
from harness.chair_gt import SYNONYMS_PATH, SYNONYMS_SHA256, ChairGT, word_tokenize

# The 80 official COCO detection category names (instances_*2014.json).
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@pytest.fixture(scope="module")
def chair():
    return ChairGT()


def test_synonym_file_is_the_reference_file():
    digest = hashlib.sha256(SYNONYMS_PATH.read_bytes()).hexdigest()
    assert digest == SYNONYMS_SHA256


def test_every_coco_category_is_a_canonical_object(chair):
    for name in COCO_CATEGORIES:
        assert chair.inverse_synonym_dict[name] == name
    assert len({chair.inverse_synonym_dict[s] for s in chair.mscoco_objects}) == 80


@pytest.mark.parametrize("word, expected", [
    ("dogs", "dog"), ("people", "person"), ("buses", "bus"), ("knives", "knife"),
    ("skis", "ski"), ("scissors", "scissors"), ("giraffes", "giraffe"),
    # reference quirks that must be preserved
    ("glass", "glas"), ("bus", "bu"),
])
def test_pattern_singularize(word, expected):
    assert singularize(word) == expected


def test_tokenizer_splits_sentences_and_punctuation():
    assert word_tokenize("a dog. a cat, sitting!") == ["a", "dog", ".", "a", "cat", ",",
                                                     "sitting", "!"]


@pytest.mark.parametrize("caption, objects", [
    ("A man riding a horse.", ["person", "horse"]),
    ("Two dogs and some people", ["dog", "person"]),
    ("A hot dog on a plate", ["hot dog"]),
    ("a baby elephant next to two men", ["elephant", "person"]),
    ("a passenger train at the station", ["train"]),
    ("a man in a bow tie", ["person", "tie"]),
    ("a wine glass on a table", ["wine glass", "dining table"]),
    ("the seat of the toilet is up", ["toilet"]),
    ("a toilet seat", ["toilet"]),
    ("a red bus", []),                  # reference quirk: singular "bus" -> "bu"
    ("two red buses", ["bus"]),
    ("a cat on a sofa next to a TV", ["cat", "couch", "tv"]),
])
def test_caption_to_objects(chair, caption, objects):
    assert chair.caption_to_words(caption)[1] == objects


def test_scores_by_hand(chair):
    chair.set_ground_truth({1: {"person", "horse"}, 2: {"cat"}, 3: {"dog"}})
    res = chair.evaluate({
        1: "a man riding a horse next to a dog",   # 3 mentions, dog hallucinated
        2: "a cat sitting on a chair near a cat",  # 3 mentions (cat twice), chair halluc.
        3: "a sunny day",                          # no COCO object mentioned
    })
    assert res.n_captions == 3 and res.n_mentions == 6 and res.n_hallucinated_mentions == 2
    assert res.chair_s == pytest.approx(2 / 3)
    assert res.chair_i == pytest.approx(2 / 6)
    assert res.captions[0].hallucinated == [("dog", "dog")]
    assert res.captions[1].hallucinated == [("chair", "chair")]


def test_unknown_image_raises(chair):
    chair.set_ground_truth({1: {"cat"}})
    with pytest.raises(KeyError):
        chair.evaluate({2: "a cat"})


def test_ground_truth_from_instances_and_reference_captions():
    instances = {
        "categories": [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"},
                       {"id": 67, "name": "dining table"}],
        "annotations": [{"image_id": 10, "category_id": 1}, {"image_id": 10, "category_id": 18},
                        {"image_id": 11, "category_id": 67}, {"image_id": 99, "category_id": 1}],
    }
    captions = {
        "images": [{"id": 10}, {"id": 11}, {"id": 99}],
        "annotations": [{"image_id": 10, "caption": "A woman walks her puppy past a bench."},
                        {"image_id": 11, "caption": "A pizza on a desk."}],
    }
    c = ChairGT.from_coco(instances, captions, image_ids=[10, 11])
    assert c.imid_to_objects == {10: {"person", "dog", "bench"},
                                 11: {"dining table", "pizza"}}
    res = c.evaluate([(10, "a man and a dog on a bench"), (10, "a girl with a frisbee"),
                      (11, "a pizza and a cup on a table")])
    assert res.n_images == 2 and res.n_captions == 3
    assert res.chair_s == pytest.approx(2 / 3)       # frisbee, cup
    assert res.chair_i == pytest.approx(2 / 8)
    counts = res.per_image_counts()
    assert counts.tolist() == [[1, 2, 1, 5], [1, 1, 1, 3]]
    s = res.summary(n_resamples=200)
    assert s["CHAIRs"]["low"] <= s["CHAIRs"]["point"] <= s["CHAIRs"]["high"]
    assert s["bootstrap"]["unit"] == "image"


def test_tokenizer_matches_nltk_word_tokenize_when_punkt_available():
    nltk = pytest.importorskip("nltk")
    text = "Two men play frisbee. A dog watches them! Is that a cat?"
    try:
        expected = nltk.word_tokenize(text)
    except LookupError:
        pytest.skip("Punkt data not installed")
    assert word_tokenize(text) == expected
