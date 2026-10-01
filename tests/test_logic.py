import pytest
import sys
import os

# Add src to python path for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from linguistic_extraction import LinguisticExtractionModule
from semantic_bridge import SemanticBridgeModule

# Initialize modules once for all tests to save time
@pytest.fixture(scope="module")
def nlp_module():
    return LinguisticExtractionModule()

@pytest.fixture(scope="module")
def semantic_module():
    return SemanticBridgeModule(threshold=0.65)


def test_linguistic_extraction_stop_words(nlp_module):
    """Test that generic scene words are filtered out."""
    caption = "A group of people taking a photo of a view."
    nouns = nlp_module.extract(caption)
    
    # 'group', 'people' (sometimes), 'photo', 'view' should be handled.
    assert "group" not in nouns
    assert "photo" not in nouns
    assert "view" not in nouns


def test_linguistic_extraction_plurals(nlp_module):
    """Test that plurals are correctly lemmatized."""
    caption = "Two dogs chase three cats near the cars."
    nouns = nlp_module.extract(caption)
    
    assert "dog" in nouns
    assert "cat" in nouns
    assert "car" in nouns
    assert "dogs" not in nouns


def test_semantic_bridge_perfect_match(semantic_module):
    """Test when DETR and Caption perfectly align."""
    detr_objects = {"person", "dog", "frisbee"}
    caption_nouns = {"person", "dog", "frisbee"}
    
    result = semantic_module.compute_alignment(detr_objects, caption_nouns)
    
    assert len(result["hallucinated"]) == 0
    assert len(result["missed"]) == 0
    assert len(result["matched"]) == 3


def test_semantic_bridge_synonyms(semantic_module):
    """Test that valid synonyms are matched and NOT hallucinated."""
    detr_objects = {"couch", "tv", "person"}
    caption_nouns = {"sofa", "television", "man"}
    
    result = semantic_module.compute_alignment(detr_objects, caption_nouns)
    
    assert len(result["hallucinated"]) == 0, f"Failed on synonyms: {result['hallucinated']}"
    assert len(result["matched"]) == 3


def test_semantic_bridge_hallucination(semantic_module):
    """Test that totally unrelated words trigger hallucination penalty."""
    detr_objects = {"person", "car"}
    caption_nouns = {"person", "dragon"}
    
    result = semantic_module.compute_alignment(detr_objects, caption_nouns)
    
    assert "dragon" in result["hallucinated"]
    assert "car" in result["missed"]


def test_semantic_bridge_empty_inputs(semantic_module):
    """Test mathematical safety (no division by zero) on empty inputs."""
    # Empty caption
    res1 = semantic_module.compute_alignment({"person"}, set())
    assert len(res1["matched"]) == 0
    
    # Empty detection
    res2 = semantic_module.compute_alignment(set(), {"person"})
    assert "person" in res2["hallucinated"]
