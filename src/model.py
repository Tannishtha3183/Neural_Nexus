"""
BioSpan AI - Neural Architecture & Inference Engine (Real PyTorch BioBERT)
Implements the full BioBERT sequence labeling pipeline:
[Raw Doctor Notes] -> [BioBERT Fast Tokenizer] -> [BioBERT Transformer Backbone]
-> [Linear Classification Head (768 -> 3 Logits)] -> [Real Softmax Confidence]
-> [First-Subword Mathematical Alignment] -> [Deterministic BIO-Sanitizer]
-> [Consolidated Entity Extractor + Negation Engine] -> [ICD-10 Ontology Linker]
"""

import os
import re
import math
import random
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.biosanitizer import BioSanitizer
from src.alignment import SubwordAligner
from src.ontology import ICD10Linker

class BioSpanEngine:
    """
    Production-Grade PyTorch BioBERT Sequence Labeling & Clinical Inference Engine.
    Executes real deep-learning tensor forward passes on clinical doctor notes.
    """

    MODEL_CHECKPOINT = "alvaroalon2/biobert_diseases_ner"

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BioSpan AI] Initializing Real BioBERT Transformer ({self.MODEL_CHECKPOINT}) on {self.device}...", flush=True)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_CHECKPOINT)
            self.model = AutoModelForTokenClassification.from_pretrained(self.MODEL_CHECKPOINT)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id
            print(f"[BioSpan AI] Model loaded successfully. Labels: {self.id2label}", flush=True)
        except Exception as e:
            print(f"[BioSpan AI] Error loading transformer weights: {e}", flush=True)
            self.is_loaded = False
            self.tokenizer = None
            self.model = None
            self.id2label = {0: "B-DISEASE", 1: "I-DISEASE", 2: "O"}

        self.sanitizer = BioSanitizer(target_entity="Disease")
        self.aligner = SubwordAligner()
        self.ontology = ICD10Linker()

        self.model_name = "BioBERT-v1.1-NCBI-Disease-NER"
        self.hidden_dim = 768
        self.attention_heads = 12
        self.transformer_layers = 12

    def tokenize_text(self, text: str) -> List[str]:
        """Splits clinical text into whitespace and punctuation-delimited words."""
        return [w for w in re.findall(r"[\w'-]+|[.,!?;:()\[\]]", text) if w.strip()]

    def _normalize_label(self, raw_label: str) -> str:
        """Maps diverse checkpoint label formats to standard BioSpan B-Disease / I-Disease / O."""
        norm = raw_label.upper().strip()
        if "B-" in norm or norm in ["B-DISEASE", "DISEASE", "B"]:
            return "B-Disease"
        elif "I-" in norm or norm in ["I-DISEASE", "I"]:
            return "I-Disease"
        return "O"

    def predict(self, text: str, simulate_orphaned_i_prob: float = 0.0) -> Dict[str, Any]:
        """
        Executes complete end-to-end BioSpan neural pipeline:
        1. Tokenizes text to words.
        2. Executes real PyTorch tensor forward pass through 12-layer BioBERT transformer.
        3. Computes real softmax probabilities across classes.
        4. Applies first-subword mathematical alignment via tokenized word_ids.
        5. Runs Deterministic BIO-Sanitizer state machine to rewire illegal transitions.
        6. Extracts clinical entities and flags ruled-out / negated mentions.
        7. Links each entity span to ICD-10 medical ontology with clinical descriptions.
        """
        words = self.tokenize_text(text)
        if not words:
            return {
                "sentence": text,
                "words": [],
                "token_count": 0,
                "raw_tags": [],
                "sanitized_tags": [],
                "raw_submission_string": "",
                "sanitized_submission_string": "",
                "confidences": [],
                "token_heatmaps": [],
                "entities": [],
                "entity_count": 0,
                "sanitizer_report": {"violation_count": 0, "repairs": [], "has_violations": False, "rescued_spans": 0},
                "subword_alignment": {"original_words": [], "token_breakdowns": []},
                "architecture_metadata": self.get_architecture_metadata()
            }

        # If model is loaded, run real PyTorch tensor forward pass
        if self.is_loaded and self.tokenizer and self.model:
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            subword_word_ids = encoding.word_ids(batch_index=0)
            subword_tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
            encoding_device = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding_device)
                logits = outputs.logits[0].cpu()  # [seq_len, num_labels]
                probs_matrix = F.softmax(logits, dim=-1)

            raw_tags = []
            confidences = []
            probabilities = []
            logits_matrix = []

            # First-subword alignment
            prev_w_id = None
            for sub_idx, w_id in enumerate(subword_word_ids):
                if w_id is None:
                    continue  # [CLS], [SEP]
                if w_id != prev_w_id:
                    # Lead subword for words[w_id]
                    sub_logits = logits[sub_idx]
                    sub_probs = probs_matrix[sub_idx]

                    pred_idx = torch.argmax(sub_logits).item()
                    raw_label = self.id2label[pred_idx]
                    norm_label = self._normalize_label(raw_label)

                    # Optional demonstration perturbation for testing sanitizer
                    if simulate_orphaned_i_prob > 0 and norm_label == "B-Disease" and random.random() < simulate_orphaned_i_prob:
                        norm_label = "I-Disease"

                    # Calculate probabilities for standard categories
                    p_b = 0.0
                    p_i = 0.0
                    p_o = 0.0
                    for c_idx, c_label in self.id2label.items():
                        c_norm = self._normalize_label(c_label)
                        val = sub_probs[c_idx].item()
                        if c_norm == "B-Disease":
                            p_b += val
                        elif c_norm == "I-Disease":
                            p_i += val
                        else:
                            p_o += val

                    prob_dict = {
                        "O": round(p_o, 4),
                        "B-Disease": round(p_b, 4),
                        "I-Disease": round(p_i, 4)
                    }
                    win_conf = round(sub_probs[pred_idx].item(), 4)

                    raw_tags.append(norm_label)
                    confidences.append(win_conf)
                    probabilities.append(prob_dict)
                    logits_matrix.append([round(x.item(), 2) for x in sub_logits])
                    prev_w_id = w_id

            # Ensure tags count exactly matches input words
            while len(raw_tags) < len(words):
                raw_tags.append("O")
                confidences.append(0.99)
                probabilities.append({"O": 0.99, "B-Disease": 0.005, "I-Disease": 0.005})
                logits_matrix.append([4.0, -2.0, -2.0])

        else:
            # Fallback path if PyTorch weights cannot be loaded
            raw_tags = ["O"] * len(words)
            confidences = [0.99] * len(words)
            probabilities = [{"O": 0.99, "B-Disease": 0.005, "I-Disease": 0.005} for _ in words]
            logits_matrix = [[4.0, -2.0, -2.0] for _ in words]

        # Step 3: Run Deterministic BIO-Sanitizer
        sanitized_res = self.sanitizer.sanitize(words, raw_tags, confidences)
        sanitized_tags = sanitized_res["sanitized_tags"]

        # Step 4: Subword Alignment Matrix
        subword_alignment = self.aligner.align_sequence(words, sanitized_tags)

        # Step 5: Consolidated Entity Extraction & Negation / Ruled-Out Detection
        spans = sanitized_res["sanitized_spans"]
        lower_text = text.lower()
        entities = []

        for span in spans:
            entity_text = span["entity_text"]
            # Negation / ruled-out heuristic detection in preceding text window
            pos = lower_text.find(entity_text.lower())
            is_ruled_out = False
            if pos != -1:
                prefix_window = lower_text[max(0, pos - 45):pos]
                negation_cues = [
                    "no history of", "denies", "ruled out", "without evidence of",
                    "negative for", "free of", "no signs of", "no prior", "no recurrent"
                ]
                if any(cue in prefix_window for cue in negation_cues):
                    is_ruled_out = True

            ontology_match = self.ontology.link_entity(entity_text)
            
            entities.append({
                "entity_text": entity_text,
                "label": "Disease",
                "start_token": span["start_token"],
                "end_token": span["end_token"],
                "token_indices": list(range(span["start_token"], span["end_token"] + 1)),
                "mean_confidence": span["mean_confidence"],
                "is_ruled_out": is_ruled_out,
                "icd10": ontology_match
            })

        # Step 6: Token heatmap generation for explainability
        token_heatmaps = []
        for idx, (word, r_tag, s_tag, conf, prob) in enumerate(zip(words, raw_tags, sanitized_tags, confidences, probabilities)):
            is_entity = s_tag != "O"
            is_sanitized_repair = (r_tag != s_tag)

            if conf >= 0.90:
                opacity = 1.0
                conf_tier = "HIGH_CONFIDENCE"
            elif conf >= 0.70:
                opacity = 0.78
                conf_tier = "MODERATE_CONFIDENCE"
            else:
                opacity = 0.52
                conf_tier = "UNCERTAIN_AMBIGUOUS"

            token_heatmaps.append({
                "index": idx,
                "word": word,
                "raw_tag": r_tag,
                "sanitized_tag": s_tag,
                "confidence": conf,
                "confidence_percent": round(conf * 100, 1),
                "confidence_tier": conf_tier,
                "opacity": opacity,
                "probabilities": prob,
                "is_entity": is_entity,
                "is_repaired": is_sanitized_repair
            })

        return {
            "sentence": text,
            "words": words,
            "token_count": len(words),
            "raw_tags": raw_tags,
            "sanitized_tags": sanitized_tags,
            "raw_submission_string": " ".join(raw_tags),
            "sanitized_submission_string": " ".join(sanitized_tags),
            "confidences": confidences,
            "token_heatmaps": token_heatmaps,
            "entities": entities,
            "entity_count": len(entities),
            "sanitizer_report": {
                "has_violations": sanitized_res["has_violations"],
                "violation_count": sanitized_res["violation_count"],
                "repairs": sanitized_res["repairs"],
                "rescued_spans": sanitized_res["span_recovery_count"]
            },
            "subword_alignment": subword_alignment,
            "architecture_metadata": self.get_architecture_metadata()
        }

    def get_architecture_metadata(self) -> Dict[str, Any]:
        return {
            "backbone": "BioBERT-v1.1 (dmis-lab/biobert-v1.1 via alvaroalon2/biobert_diseases_ner)",
            "device": str(self.device).upper(),
            "status": "LIVE_PYTORCH_EVAL_MODE" if self.is_loaded else "FALLBACK_MODE",
            "hidden_dimension": self.hidden_dim,
            "attention_heads": self.attention_heads,
            "transformer_layers": self.transformer_layers,
            "parameters": "108 Million Transformer Parameters",
            "training_dataset": "NCBI Disease Corpus (Hugging Face / PubMed)",
            "loss_masking": "ignore_index=-100",
            "alignment": "First-Subword Mathematical Alignment",
            "post_processing": "Deterministic BIO-Sanitizer Finite-State Machine"
        }
