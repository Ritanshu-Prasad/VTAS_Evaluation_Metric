# 👁️ VTAS: Visual-Truth Alignment Score
**A Novel Multi-Model Evaluation Framework for Image Captioning**

## The Core Problem
Traditional n-gram metrics (BLEU, METEOR, CIDEr) are "blind". They compare generated text to human reference text without ever looking at the actual image. 
* This leads to heavy penalties for valid synonyms.
* It fails to effectively penalize visual hallucinations if the ground truth reference is flawed or incomplete.

## The Solution
A custom evaluation metric engineered to directly verify the generated text against the physical pixels of the image, rather than relying solely on human text annotations.

---

## 🏗️ Architecture & Pipeline (Compound AI System)

### 1. Visual Grounding Module (The "Eyes")
- **Technology:** Lightweight, pre-trained object detection pipeline (e.g., `facebook/detr-resnet-50` via Hugging Face `transformers`).
- **Input:** Raw Image.
- **Output:** Set of detected physical objects (e.g., `Set(V) = {'person', 'dog', 'frisbee'}`).

### 2. Linguistic Extraction Module (The "Brain")
- **Technology:** Natural Language Toolkit (`nltk`) or `spaCy`.
- **Action:** Parse the Vision-Language Model's generated caption using Part-of-Speech (POS) tagging to extract primary nouns and objects.
- **Input:** Generated string (e.g., *"A man throws a frisbee to his cat"*).
- **Output:** Set of textual objects (e.g., `Set(T) = {'man', 'frisbee', 'cat'}`).

### 3. Scoring Engine (The Math)
- **Semantic Bridging (Resolving Vocabulary Mismatch):** Utilize sentence embeddings like `all-MiniLM-L6-v2` to calculate cosine similarity between DETR objects and text nouns. (e.g. `cosine("couch", "sofa") > 0.75` counts as a match, preventing false hallucination penalties).
- **Visual Recall (Intersection):** Calculate Semantic Overlap. Did the text mention the key objects physically present in the image?
- **Hallucination Penalty (Difference):** Did the text invent objects that the detector confirms do not exist in the image? (e.g., text says 'cat', but detector found no cat = heavy penalty).
- **Final Calculation:** Combine Recall and the Hallucination Penalty into a single continuous score (0 to 1) representing true multimodal alignment.

---

## 🌍 Industry Landscape & Novelty
When defending VTAS in an academic or professional setting, it is critical to position it against the current State-of-the-Art (SOTA) evaluation metrics to demonstrate its unique engineering value.

### Existing Approaches (The Baseline)
1. **CLIPScore (VTAS v2 equivalent):** The current industry standard for reference-free evaluation. It embeds both image and text to find cosine similarity.
   - *The Flaw:* It is a "black box". It outputs a score but cannot provide interpretability on *why* the score is low (e.g., did the model miss an object, or hallucinate one?).
2. **LLM-as-a-Judge (VTAS v3 equivalent):** Passing the image and caption to a frontier model (like GPT-4V) to grade hallucination. 
   - *The Flaw:* High API costs, slow latency, and reliance on closed-source models.
3. **CHAIR (Caption Hallucination Assessment with Image Relevance):** A metric specifically designed to detect object hallucinations.
   - *The Flaw:* It requires expensive, human-annotated ground-truth object tags. It cannot run automatically on raw, unannotated images.

### The VTAS v1 Novelty
VTAS v1 is genuinely novel because of its **architectural pipeline**. It acts as an automated, white-box Compound AI System using modern transformers to solve the flaws of the existing metrics:
- **White-Box Interpretability:** Unlike CLIPScore, VTAS explicitly extracts objects and can pinpoint exactly *which* word triggered a hallucination penalty.
- **Fully Automated Ground Truth:** Unlike CHAIR, VTAS does not need human annotators. It uses a live Object Detector (DETR) as a mechanical eye to establish the ground truth dynamically.
- **Semantic Robustness:** Unlike traditional NLP n-gram matching, VTAS uses sentence embeddings (`all-MiniLM-L6-v2`) as a semantic bridge, solving the vocabulary mismatch problem where "sofa" and "couch" would previously trigger false hallucination penalties.

---

## 🚀 Implementation Strategy
Develop VTAS as a heavily documented standalone framework. 
Provide multiple concrete examples of how it handles edge cases that traditional metrics fail on:
1. **The Synonym Edge Case:** Where the model is correct but uses different vocabulary than the ground truth.
2. **The Missing Reference Edge Case:** Where the model correctly identifies a background object that the human annotator forgot to mention.
3. **The Confident Hallucination:** Where the model describes something highly likely but factually absent from the image.

---

## 🔬 Future Work & Optimization
- **Dynamic Thresholding:** Eliminate the hardcoded `0.75` cosine similarity cutoff by researching dynamic thresholds or using a lightweight classifier to determine valid semantic matches.
- **Compute Optimization:** Solve the massive computational overhead of running three separate models (BLIP, DETR, MiniLM) by exploring unified multimodal embeddings (like CLIP) to perform text-to-vision grounding in a single, lightweight forward pass.
