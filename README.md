# 🧬 BioSpan AI: Biomedical Named Entity Recognition (NER) & Clinical Ontology Engine

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-BioBERT%20v1.1-yellow.svg)](https://huggingface.co/dmis-lab/biobert-v1.1)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **BioSpan AI** is an advanced Biomedical Named Entity Recognition (NER) and Clinical Ontology Linker system engineered specifically to resolve the core challenges of unstructured electronic medical records: multi-word phenotype boundary detection, subword shredding distortion, and unconstrained decoder hallucinations.

---

## 📑 Table of Contents
1. [System Overview & Core Philosophy](#1-system-overview--core-philosophy)
2. [The Three Architectural Pillars](#2-the-three-architectural-pillars)
3. [The Data Layer & BIO Tagging Mechanics](#3-the-data-layer--bio-tagging-mechanics)
4. [Tokenization Dilemma & Subword Mathematical Alignment](#4-tokenization-dilemma--subword-mathematical-alignment)
5. [Neural Architecture & Representation Learning](#5-neural-architecture--representation-learning)
6. [The Deterministic BIO-Sanitizer (The Key Innovation)](#6-the-deterministic-bio-sanitizer-the-key-innovation)
7. [Clinical Ontology Linker (ICD-10 Integration)](#7-clinical-ontology-linker-icd-10-integration)
8. [Evaluation & Defense Masterclass](#8-evaluation--defense-masterclass)
9. [Full-Stack Web Interface & Features](#9-full-stack-web-interface--features)
10. [Quickstart & Usage Instructions](#10-quickstart--usage-instructions)

---

## 1. System Overview & Core Philosophy
The primary objective of **BioSpan AI** is solving **Biomedical Named Entity Recognition (NER)**: taking raw, unstructured medical text and accurately locating the exact start and end boundaries of disease and phenotype mentions.

Standard NLP approaches fail in the medical domain because disease names are rarely single words (e.g., *"non-small cell lung carcinoma"*, *"autoimmune hemolytic anemia"*). If an algorithm flags only *"lung carcinoma"*, it commits a critical clinical error by discarding the phenotype modifier that dictates patient staging and oncological treatment protocols.

### 📚 Authentic Benchmark Datasets (Hugging Face / Google Scholar / NCBI)
BioSpan AI is evaluated and integrated with the gold-standard **NCBI Disease Corpus**:
- **Hugging Face Hub:** [`ncbi/ncbi_disease`](https://huggingface.co/datasets/ncbi/ncbi_disease)
- **Academic Citation (Google Scholar & PubMed):**
  > Doğan, R. I., Leaman, R., & Lu, Z. (2014). *NCBI disease corpus: a resource for disease name recognition and concept normalization*. Journal of Biomedical Informatics, 47, 1-10.
- **Dataset Files Included in Repository:**
  - [`data/test.tsv`](file:///c:/Users/Nishtha/OneDrive/Desktop/Projects/Neural%20Nexus/data/test.tsv): Authentic 25,437-token CoNLL test corpus from NCBI/PubMed.
  - [`data/devel.tsv`](file:///c:/Users/Nishtha/OneDrive/Desktop/Projects/Neural%20Nexus/data/devel.tsv): Authentic CoNLL validation corpus.
  - [`data/ncbi_disease_test.csv`](file:///c:/Users/Nishtha/OneDrive/Desktop/Projects/Neural%20Nexus/data/ncbi_disease_test.csv): 940 peer-reviewed test sentences formatted for batch submission (`sentence_id,text`).
  - [`data/ncbi_disease_gold.csv`](file:///c:/Users/Nishtha/OneDrive/Desktop/Projects/Neural%20Nexus/data/ncbi_disease_gold.csv): Full ground-truth reference tags and multi-word entity spans.
  - [`data/ncbi_disease_cases.json`](file:///c:/Users/Nishtha/OneDrive/Desktop/Projects/Neural%20Nexus/data/ncbi_disease_cases.json): Curated clinical case studies from PubMed for 1-click evaluation.


## 2. The Three Architectural Pillars

```
                     ┌──────────────────────────────────────────────┐
                     │                 BioSpan AI                   │
                     └──────────────────────┬───────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                   ▼                                   ▼
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│     Pillar 1:         │       │     Pillar 2:         │       │     Pillar 3:         │
│  Domain-Specific      │       │  First-Subword        │       │  Deterministic        │
│  BioBERT Embeddings   │       │  Mathematical         │       │  Output Constraining  │
│                       │       │  Alignment (-100 Mask)│       │  (The BIO-Sanitizer)  │
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

1. **Domain-Specific Semantic Representation:** Utilizing BioBERT, whose contextual embeddings were pre-trained on billions of tokens from PubMed biomedical abstracts and PMC full-text articles to preserve medical vocabulary and semantic neighborhoods.
2. **First-Subword Mathematical Alignment:** Resolving the tokenization mismatch between transformer subwords and real-world clinical vocabulary by selectively masking continuation fragments with `-100`.
3. **Deterministic Output Constraining (The BIO-Sanitizer):** Enforcing the grammatical rules of the BIO tagging scheme during post-processing to eliminate the statistical hallucinations common in unconstrained neural decoders.

---

## 3. The Data Layer & BIO Tagging Mechanics

### Why BIO Tagging is Mandatory
If you use a simple binary tag (*Disease* vs. *Not-Disease*), the model encounters an insurmountable ambiguity whenever two distinct disease mentions appear adjacent to one another:

> *"The patient presented with **asthma**, **bronchitis**, and **fever**."*

A binary classifier flags both as positive entities, leaving the downstream system unable to determine whether the patient has one composite disease or two distinct conditions.

**BIO tagging introduces directional topology to the text:**
- **`O` (Outside):** Tokens carrying no disease meaning in this context.
- **`B-Disease` (Begin):** The mandatory first token of any disease mention.
- **`I-Disease` (Inside):** Continuation tokens strictly belonging to the same disease mention that began with an immediately preceding `B-Disease` or `I-Disease`.

### The Class Imbalance Problem
In biomedical corpora of ~5,000 sentences, **85% to 92% of all tokens are tagged as `O`**. A naive model predicting `O` everywhere achieves an impressive **90% accuracy** while finding **zero diseases**. BioSpan AI bypasses this trap by enforcing strict **Exact Entity Span F1** metric checkpointing.

---

## 4. Tokenization Dilemma & Subword Mathematical Alignment

### The Problem of Subword Shredding
When a complex medical term like `"Cardiomyopathy"` enters a standard WordPiece tokenizer, it is shredded into subword fragments:

$$\text{"Cardiomyopathy"} \longrightarrow \text{["Cardio", "\#\#myo", "\#\#pathy"]}$$

### The First-Subword Alignment Protocol
BioSpan AI resolves this through **First-Subword Labeling with Loss-Masking**:
1. **The Lead Fragment:** The first subword (`"Cardio"`) is assigned the ground-truth label (`B-Disease`). This fragment carries the forward context and acts as the anchor.
2. **Continuation Fragments:** All subsequent subwords (`"##myo"`, `"##pathy"`) are assigned the special numerical label `-100`.
3. **Special Sentence Tokens:** `[CLS]` and `[SEP]` are also assigned `-100`.

### Why `-100` Matters
PyTorch’s cross-entropy loss function is mathematically configured to ignore index `-100` during backpropagation (`torch.nn.CrossEntropyLoss(ignore_index=-100)`). 
- If continuation fragments were also tagged, words splitting into 5 pieces would exert **5x more gradient pull** than single-token words like *"asthma"*, severely distorting model parameter updates.
- By masking continuation fragments, the model is penalized and rewarded **exactly once per clinical word**.

---

## 5. Neural Architecture & Representation Learning

BioSpan AI runs **genuine PyTorch deep learning transformer inference** using the fine-tuned BioBERT checkpoint (`alvaroalon2/biobert_diseases_ner`), trained on the gold-standard NCBI Disease Corpus:

```
[Raw Doctor Note / Unstructured Medical Text]
          │
          ▼
[BioBERT Fast Tokenizer] ───► WordPiece tokenization with subword tracking & word_ids
          │
          ▼
[BioBERT Transformer Backbone] ───► 12-layer bidirectional transformer (108M parameters, 768-dim, 12 attention heads)
          │
          ▼
[Linear Token Classification Head] ───► Projects 768-dim embeddings to 3 BIO logits (O, B-DISEASE, I-DISEASE)
          │
          ▼
[PyTorch Softmax] ───► Computes exact calibrated posterior probabilities per subword
          │
          ▼
[First-Subword Mathematical Alignment] ───► Preserves lead subword predictions, maps back to original clinician words
          │
          ▼
[Deterministic BIO-Sanitizer] ───► Rewires illegal syntax transitions (O ──► I-Disease becomes O ──► B-Disease)
          │
          ▼
[Clinical Negation & Rule-Out Engine] ───► Detects negated phrases ("no history of", "denies", "ruled out")
          │
          ▼
[Clinical Ontology Linker (ICD-10)] ───► Maps extracted multi-word spans to official ICD-10 codes, categories & plain English notes
```

- **True Neural Inference**: Every doctor note processed by BioSpan AI undergoes a real PyTorch forward tensor pass with real softmax confidence scores.
- **Zero Simulation / No Heuristic Guessing**: Spans are detected directly by the self-attention heads of BioBERT pre-trained on PubMed and fine-tuned on clinical disease mentions.
- **Generalization**: Accurately recognizes arbitrary doctor notes containing complex multi-word oncology mentions (*"non-small cell lung cancer"*), metabolic disorders (*"type 2 diabetes mellitus"*), cardiology events (*"myocardial infarction"*), and neurological symptoms (*"cerebral edema"*, *"seizures"*, *"headache"*).

---

## 6. The Deterministic BIO-Sanitizer (The Key Innovation)

Transformers classify tokens independently at inference time and lack an explicit grammar state machine. Consequently, unconstrained decoders occasionally emit **orphaned `I-Disease` tags**:

$$\text{Input Words: } [\text{"Patient"}, \text{"denies"}, \text{"chest"}, \text{"pain"}]$$
$$\text{Raw Model Output: } [\text{"O"}, \text{"O"}, \text{"O"}, \text{"I-Disease"}]$$

An `I-Disease` tag cannot exist without an initiating `B-Disease` tag. Under exact span evaluation, this orphaned `I-Disease` causes a **double penalty**:
1. **False Negative:** Fails to recognize the true disease mention.
2. **False Positive:** Introduces an invalid hallucinated span.

### The Sanitizer State Machine
The **BIO-Sanitizer** parses the predicted sequence token by token:
- If it encounters an `I-Disease` tag when the preceding token is `O` (or at index 0), the sanitizer intercepts the violation and **dynamically rewires it into a `B-Disease` tag**.
- This restores grammatical legality, rescues the entity boundary, and preserves Exact Span F1 without requiring retraining.

---

## 7. Clinical Ontology Linker (ICD-10 Integration)

Extracting spans is only the first step in clinical applications; hospitals require standardized diagnostic codes for Electronic Health Records (EHR) and billing.

BioSpan AI routes every isolated entity span through an **Ontological Mapping Service**:
- **Non-small cell lung carcinoma** $\rightarrow$ `C34.90` (*Malignant Neoplasms of Respiratory Organs*)
- **Dilated cardiomyopathy** $\rightarrow$ `I42.0` (*Diseases of the Circulatory System*)
- **Acute myocardial infarction** $\rightarrow$ `I21.9` (*Emergency / Critical Care DRG*)
- **Autoimmune hemolytic anemia** $\rightarrow$ `D59.10` (*Hematology Specialist*)
- **Severe persistent asthma** $\rightarrow$ `J45.909` (*Chronic Lower Respiratory Diseases*)

---

## 8. Evaluation & Defense Masterclass

| Metric | Calculation Method | Clinical Reality | Flaw / Verdict |
| :--- | :--- | :--- | :--- |
| **Token Accuracy** | $\frac{\text{Correct Tokens}}{\text{Total Tokens}}$ | Dominated by $\sim 90\%$ `O` tokens | **Completely Fake**: Guessing `O` everywhere scores 90% while detecting zero diseases. |
| **Token-Level F1** | Token-by-token Precision & Recall | Rewards partial word overlap | **Deceptive**: Predicting *"lung cancer"* instead of *"non-small cell lung cancer"* gets 40% partial credit despite discarding phenotype modifier. |
| **Exact Span F1** | Strict match on `(Start, End, Label)` | Requires exact multi-word boundaries | **The Clinical Gold Standard**: Omitting a modifier produces both 1 False Negative and 1 False Positive. |

---

## 9. Full-Stack Web Interface & Design System

The BioSpan AI web application strictly adheres to the **Stitch Design System** specified in [`code.html`](file:///c:/Users/Nishtha/OneDrive/Desktop/Projects/Neural%20Nexus/code.html), [`screen.png`](file:///c:/Users/Nishtha/OneDrive/Desktop/Projects/Neural%20Nexus/screen.png), and [`DESIGN.md`](file:///c:/Users/Nishtha/OneDrive/Desktop/Projects/Neural%20Nexus/DESIGN.md):

- **Aesthetic Philosophy — "Sahara: Warm Minimalism":**
  - **Color Palette:** Warm linen background (`#faf5ee`), burnt sienna primary accents (`#c2652a`), dusty rose tertiary highlights (`#8c3c3c`), and warm grays.
  - **Typography:** Editorial serif headlines in **EB Garamond** paired with modern geometric body text in **Manrope**.
  - **Elevation:** Ultra-soft atmospheric drop shadows (`0 2px 16px rgba(58, 48, 42, 0.04)`), frosted glass headers, and warm tinting.
- **Interactive Workspace & Features:**
  - **Two-Column Note & Findings Workspace:** Left side provides text input with word/paragraph counting, quick preset buttons (Lung, Diabetes, Heart, NCBI PubMed Benchmark), and one-click analysis. Right side renders live interactive entity pills with active/secondary/ruled-out tags.
  - **Entity Detail Card:** Displays selected condition name, ICD-10 code, clinical category, and plain English explanation with flash focus animations.
  - **Plain English Summary Table:** Structured findings table with clinical statuses, plain translations, and next steps for patients and physicians.
  - **Action Controls:** One-click copy for individual terms, complete clipboard summaries, and structured clinical brief downloads.
  - **Clinical ICD-10 Dictionary Modal:** Accessible directly from navigation and cards to look up codes, synonyms, and DRG classifications.
  - **Saved Notes:** Local storage-backed session history for clinical consultations.
- **RESTful Endpoints:**
  - `POST /api/predict`: Real-time BioBERT sequence labeling, subword alignment, BIO-sanitization, and ICD-10 linking.
  - `GET /api/ontology`: Searchable ICD-10 disease dictionary.
  - `GET /api/dataset/ncbi-test-batch`: Real NCBI Disease Corpus evaluation cases.
  - `GET /api/dataset/download-ncbi-csv`: Downloadable NCBI Disease test benchmark CSV.

---

## 10. Quickstart & Usage Instructions

### Running with One Click (Windows)
Double-click `run_project.bat` in the project root directory. It will:
1. Validate Python 3.10.
2. Start the FastAPI server on `http://127.0.0.1:8000`.
3. Automatically open the BioSpan AI dashboard in your default browser.

### Running Manually via Command Line
```powershell
# From the project root:
py -3.10 server.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### Running Interactive Tests / Verification
```powershell
py -3.10 -c "from src.model import BioSpanEngine; engine = BioSpanEngine(); print(engine.predict('Patient diagnosed with dilated cardiomyopathy and severe asthma.'))"
```

---
*Developed with BioBERT v1.1 Backbone, First-Subword Alignment Protocol, and Deterministic BIO-Sanitization.*
