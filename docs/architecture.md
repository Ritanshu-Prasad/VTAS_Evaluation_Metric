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
- **Output:** Set of detected physical objects with confidence scores (e.g., `Set(V) = {'person', 'dog', 'frisbee'}`).

### 2. Linguistic Extraction Module (The "Brain")
- **Technology:** Natural Language Toolkit (`nltk`) or `spaCy`.
- **Action:** Parse the Vision-Language Model's generated caption using Part-of-Speech (POS) tagging to extract primary nouns and objects.
- **Input:** Generated string (e.g., *"A man throws a frisbee to his cat"*).
- **Output:** Set of textual objects (e.g., `Set(T) = {'man', 'frisbee', 'cat'}`).

### 3. Semantic Bridging Module (The "Translator")
- **Technology:** Sentence embeddings via `all-MiniLM-L6-v2`.
- **Action:** For every pair `(t, v)` where `t ∈ Set(T)` and `v ∈ Set(V)`, compute `cosine_similarity(embed(t), embed(v))`.
- **Purpose:** Resolves vocabulary mismatch so that "sofa" and "couch" are recognized as the same concept.

### 4. Scoring Engine (The Math)
See **Mathematical Formulation** section below for the exact equations.

---

## 📐 Mathematical Formulation

Given:
- $V = \{v_1, v_2, ..., v_n\}$ — Set of objects detected by DETR in the image
- $T = \{t_1, t_2, ..., t_m\}$ — Set of nouns extracted from the generated caption
- $\text{sim}(t_i, v_j) = \text{cosine}(\text{embed}(t_i), \text{embed}(v_j))$ — Semantic similarity between a text noun and a detected object
- $\tau$ — Similarity threshold (default: 0.75)

### Step 1: Semantic Matching
A text noun $t_i$ is considered a **valid match** if there exists at least one detected object $v_j$ such that:

$$\text{matched}(t_i) = \begin{cases} 1 & \text{if } \max_{v_j \in V} \text{sim}(t_i, v_j) \geq \tau \\ 0 & \text{otherwise (hallucination)} \end{cases}$$

### Step 2: Visual Recall (VR)
Measures how many of the detected objects were mentioned in the caption:

$$VR = \frac{|\{v_j \in V : \max_{t_i \in T} \text{sim}(t_i, v_j) \geq \tau\}|}{|V|}$$

### Step 3: Hallucination Rate (HR)
Measures how many of the text nouns have no grounding in the image:

$$HR = \frac{|\{t_i \in T : \text{matched}(t_i) = 0\}|}{|T|}$$

### Step 4: Final VTAS Score

$$\text{VTAS} = VR \times (1 - HR)$$

- **Score = 1.0:** The caption mentioned every detected object and hallucinated nothing. Perfect alignment.
- **Score = 0.0:** The caption either missed everything or hallucinated everything.
- **Range:** Continuous value in $[0, 1]$.

---

## 🔍 Worked Examples

### Example 1: The Perfect Caption
**Image:** A park scene with a person throwing a frisbee to a dog.

| Module | Output |
|--------|--------|
| **DETR** | `Set(V) = {'person', 'dog', 'frisbee'}` |
| **NLP** | Caption: *"A man throws a frisbee to his dog"* → `Set(T) = {'man', 'frisbee', 'dog'}` |
| **Bridging** | `sim('man', 'person') = 0.82 ✅`, `sim('frisbee', 'frisbee') = 1.0 ✅`, `sim('dog', 'dog') = 1.0 ✅` |
| **VR** | 3/3 = **1.0** |
| **HR** | 0/3 = **0.0** |
| **VTAS** | 1.0 × (1 - 0.0) = **1.0** ✅ Perfect |

### Example 2: The Synonym Edge Case (Where BLEU Fails, VTAS Succeeds)
**Image:** A living room with a person sitting on a couch watching TV.

| Module | Output |
|--------|--------|
| **DETR** | `Set(V) = {'person', 'couch', 'tv'}` |
| **NLP** | Caption: *"A man relaxes on a sofa watching television"* → `Set(T) = {'man', 'sofa', 'television'}` |
| **Bridging** | `sim('man', 'person') = 0.82 ✅`, `sim('sofa', 'couch') = 0.89 ✅`, `sim('television', 'tv') = 0.91 ✅` |
| **VR** | 3/3 = **1.0** |
| **HR** | 0/3 = **0.0** |
| **VTAS** | 1.0 × (1 - 0.0) = **1.0** ✅ Perfect |

> **Note:** BLEU-4 would score this very low because none of the exact 4-grams match the reference. VTAS correctly scores it as perfect because the semantic meaning is identical.

### Example 3: The Confident Hallucination (Where Traditional Metrics Fail)
**Image:** A boy holding an umbrella standing near a cow.

| Module | Output |
|--------|--------|
| **DETR** | `Set(V) = {'person', 'umbrella', 'cow'}` |
| **NLP** | Caption: *"A group of people standing with umbrellas and dogs"* → `Set(T) = {'group', 'people', 'umbrellas', 'dogs'}` |
| **Bridging** | `sim('people', 'person') = 0.85 ✅`, `sim('umbrellas', 'umbrella') = 0.97 ✅`, `sim('group', ...) < 0.75 ❌`, `sim('dogs', 'cow') = 0.45 ❌` |
| **VR** | 2/3 = **0.67** (missed 'cow') |
| **HR** | 2/4 = **0.50** (hallucinated 'group' and 'dogs') |
| **VTAS** | 0.67 × (1 - 0.50) = **0.33** ❌ Low — correctly penalized |

> **Note:** This is the exact hallucination issue we discovered in our BLIP zero-shot baseline during the TPA-19 project. Traditional metrics might still score this reasonably if the reference caption happens to mention "people." VTAS catches the hallucination because DETR physically confirms there is no "group" or "dogs" in the image.

### Example 4: The Missing Reference Edge Case
**Image:** A busy street with a car, a person, and a traffic light in the background.

| Module | Output |
|--------|--------|
| **DETR** | `Set(V) = {'car', 'person', 'traffic light'}` |
| **NLP** | Caption: *"A person walks past a car near a traffic light"* → `Set(T) = {'person', 'car', 'traffic light'}` |
| **VR** | 3/3 = **1.0** |
| **HR** | 0/3 = **0.0** |
| **VTAS** | 1.0 × (1 - 0.0) = **1.0** ✅ Perfect |

> **Note:** If the human annotator's reference caption only said *"A person walks on the street"* (forgetting to mention the car and traffic light), BLEU would penalize this caption for mentioning extra objects. VTAS correctly rewards it because those objects are physically present in the image.

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

## 🗺️ Versioning Roadmap

### VTAS v1: The Explicit / Heavyweight Metric (Current)
- **Pipeline:** DETR (Vision) + spaCy/NLTK (NLP Extraction) + MiniLM (Semantic Bridging)
- **Tradeoff:** Computationally heavy (3 models in memory), but highly **interpretable**. You can print exactly which word triggered a penalty.
- **Best For:** Debugging and development. Understanding *why* a model is failing.

### VTAS v2: The Implicit / Lightweight Metric
- **Pipeline:** CLIP only. Encode image and text into a shared embedding space. Compute cosine similarity.
- **Tradeoff:** Extremely fast and lightweight, but a complete "black box". No interpretability on failure cases.
- **Best For:** Production environments. Scoring millions of images at scale where speed matters more than debugging.
- **Note:** This is functionally equivalent to CLIPScore but reframed within the VTAS framework for consistent benchmarking.

### VTAS v3: LLM-as-a-Judge
- **Pipeline:** Pass the image + caption to a frontier multimodal model (GPT-4o, Gemini Pro) with a structured grading prompt.
- **Tradeoff:** Near-human reasoning and catches complex spatial/relational errors, but costs money (API tokens) and depends on closed-source models.
- **Best For:** Final quality audits. Catching subtle errors that v1 and v2 miss (e.g., "the man is *behind* the car" vs. "the man is *inside* the car").

### Comparison Matrix

| Property | VTAS v1 | VTAS v2 | VTAS v3 |
|----------|---------|---------|---------|
| **Speed** | Slow | Very Fast | Slow |
| **Cost** | Free (local) | Free (local) | Paid (API) |
| **Interpretability** | Full (white-box) | None (black-box) | Partial (text explanation) |
| **Spatial Reasoning** | None | Weak | Strong |
| **Hallucination Detection** | Strong | Moderate | Very Strong |
| **Open Source** | ✅ Yes | ✅ Yes | ❌ No |

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
- **Spatial Awareness (v1.1):** Extend VTAS v1 to incorporate bounding box positions from DETR, enabling it to penalize spatial errors (e.g., "on" vs. "under" vs. "behind").
- **Attribute-Level Grounding:** Beyond object detection, incorporate color and texture classifiers to verify attribute-level claims (e.g., "red car" vs. "blue car").
