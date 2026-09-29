"""
BioSpan AI - First-Subword Mathematical Alignment Engine
Resolves tokenization mismatch between transformer subwords and clinical vocabulary.
Implements First-Subword Labeling with Loss-Masking (-100).
"""

from typing import List, Dict, Tuple, Any
import re

class SubwordAligner:
    """
    Subword Shredding and Mathematical Alignment Engine.
    
    Demonstrates and executes:
    1. WordPiece Tokenization into subword fragments.
    2. Lead-fragment label assignment.
    3. Continuation-fragment loss masking with -100.
    4. Special token masking ([CLS], [SEP]).
    5. Gradient flow dynamics (1x balanced pull vs naive 5x distortion).
    6. Reconstruction of original word boundaries via word_ids mapping.
    """

    # Clinical vocabulary subword shredding patterns for realistic demonstration
    KNOWN_SUBWORDS = {
        "cardiomyopathy": ["Cardio", "##myo", "##pathy"],
        "hypercholesterolemia": ["Hyper", "##cho", "##les", "##terol", "##emia"],
        "carcinoma": ["carcin", "##oma"],
        "adenocarcinoma": ["adeno", "##carcin", "##oma"],
        "hemolytic": ["hemo", "##lytic"],
        "bronchitis": ["bronch", "##itis"],
        "atherosclerosis": ["ath", "##ero", "##sclero", "##sis"],
        "thrombocytopenia": ["thromb", "##o", "##cyto", "##penia"],
        "encephalopathy": ["enceph", "##alo", "##pathy"],
        "glioblastoma": ["glio", "##blast", "##oma"],
        "lymphadenopathy": ["lymph", "##adeno", "##pathy"],
        "retinopathy": ["retin", "##o", "##pathy"],
        "nephropathy": ["nephro", "##pathy"],
        "osteoarthritis": ["osteo", "##arthr", "##itis"],
        "pneumonitis": ["pneumon", "##itis"],
        "hemoptysis": ["hemo", "##pty", "##sis"],
        "myocardial": ["myo", "##cardial"],
        "infarction": ["in", "##farc", "##tion"],
        "fibrosis": ["fibro", "##sis"],
        "neuropathy": ["neuro", "##pathy"]
    }

    def __init__(self):
        pass

    def simulate_tokenization(self, word: str) -> List[str]:
        """Splits a single clinical word into subwords using WordPiece heuristics."""
        clean = re.sub(r'[^\w]', '', word.lower())
        if clean in self.KNOWN_SUBWORDS:
            return self.KNOWN_SUBWORDS[clean]
        
        # WordPiece morphological splitting fallback
        if len(clean) > 8:
            if clean.endswith("emia"):
                return [clean[:-4], "##emia"]
            if clean.endswith("itis"):
                return [clean[:-4], "##itis"]
            if clean.endswith("oma"):
                return [clean[:-3], "##oma"]
            if clean.endswith("pathy"):
                return [clean[:-5], "##pathy"]
            if clean.startswith("hyper") and len(clean) > 9:
                return ["hyper", "##" + clean[5:]]
            if clean.startswith("auto") and len(clean) > 8:
                return ["auto", "##" + clean[4:]]
            # Arbitrary 2-piece split for long unfamiliar medical compound
            mid = len(clean) // 2
            return [clean[:mid], "##" + clean[mid:]]
        
        return [word]

    def align_sequence(self, words: List[str], gold_tags: List[str] = None) -> Dict[str, Any]:
        """
        Processes a sequence of words through the First-Subword Alignment Protocol:
        - Injects [CLS] at position 0 (label: -100)
        - For each word:
            - Subword 0 (Lead fragment): Gets the gold tag (e.g. B-Disease), mask = 1 (active gradient)
            - Subwords 1..K (Continuation): Gets label -100, mask = 0 (loss ignored)
        - Injects [SEP] at the end (label: -100)
        """
        if gold_tags is None:
            gold_tags = ["O"] * len(words)

        subwords = ["[CLS]"]
        word_ids = [None]
        aligned_labels = [-100]
        aligned_tags_str = ["-100"]
        loss_masks = [0] # 0 = masked (-100), 1 = active
        fragment_types = ["SPECIAL_TOKEN"]
        
        token_breakdowns = []
        naive_labels = ["O"] # comparison for naive labeling distortion
        
        tag2id = {"O": 0, "B-Disease": 1, "I-Disease": 2}
        id2tag = {0: "O", 1: "B-Disease", 2: "I-Disease", -100: "-100"}

        for word_idx, (word, gold_tag) in enumerate(zip(words, gold_tags)):
            frags = self.simulate_tokenization(word)
            word_subwords_data = []

            for frag_idx, frag in enumerate(frags):
                subwords.append(frag)
                word_ids.append(word_idx)
                
                if frag_idx == 0:
                    # Lead Fragment - The Anchor
                    tag_id = tag2id.get(gold_tag, 0)
                    aligned_labels.append(tag_id)
                    aligned_tags_str.append(gold_tag)
                    loss_masks.append(1)
                    frag_type = "LEAD_FRAGMENT (ANCHOR)"
                    frag_desc = f"Carries gold label '{gold_tag}'. Forward context anchor. Included in gradient backpropagation."
                else:
                    # Continuation Fragment - Masked with -100
                    aligned_labels.append(-100)
                    aligned_tags_str.append("-100")
                    loss_masks.append(0)
                    frag_type = "CONTINUATION_FRAGMENT (MASKED)"
                    frag_desc = "Continuation piece. Masked to -100. Ignored by PyTorch cross_entropy loss."

                fragment_types.append(frag_type)
                
                # In naive approach, continuation fragments get duplicate gold tags
                naive_labels.append(gold_tag)
                
                word_subwords_data.append({
                    "fragment": frag,
                    "fragment_index": frag_idx,
                    "is_lead": (frag_idx == 0),
                    "aligned_tag": gold_tag if frag_idx == 0 else "-100",
                    "loss_mask": 1 if frag_idx == 0 else 0,
                    "gradient_weight": "1.0x" if frag_idx == 0 else "0.0x (Masked)",
                    "naive_gradient_weight": "1.0x (DISTORTED)" if frag_idx > 0 else "1.0x"
                })

            token_breakdowns.append({
                "word_index": word_idx,
                "original_word": word,
                "gold_tag": gold_tag,
                "subword_count": len(frags),
                "is_shredded": len(frags) > 1,
                "subwords": word_subwords_data
            })

        # Add [SEP]
        subwords.append("[SEP]")
        word_ids.append(None)
        aligned_labels.append(-100)
        aligned_tags_str.append("-100")
        loss_masks.append(0)
        fragment_types.append("SPECIAL_TOKEN")
        naive_labels.append("O")

        # Mathematical comparison of gradient balance
        active_tokens = sum(loss_masks)
        total_subwords = len(subwords)
        naive_active_tokens = len(words) + sum(len(b["subwords"]) - 1 for b in token_breakdowns)
        distortion_percentage = round(((naive_active_tokens - len(words)) / len(words)) * 100, 1) if len(words) > 0 else 0.0

        return {
            "original_words": words,
            "word_count": len(words),
            "subwords": subwords,
            "subword_count": total_subwords,
            "word_ids": word_ids,
            "aligned_labels": aligned_labels,
            "aligned_tags_str": aligned_tags_str,
            "loss_masks": loss_masks,
            "active_tokens": active_tokens,
            "token_breakdowns": token_breakdowns,
            "gradient_balance": {
                "active_tokens_biospan": active_tokens,
                "active_tokens_naive": naive_active_tokens,
                "distortion_avoided_pct": f"+{distortion_percentage}%",
                "loss_function_setting": "torch.nn.CrossEntropyLoss(ignore_index=-100)",
                "mathematical_guarantee": "Each clinical word exerts exactly 1.0x gradient pull regardless of subword shredding count."
            }
        }

    def reconstruct_from_predictions(self, words: List[str], word_ids: List[int], 
                                    subword_predictions: List[str]) -> List[str]:
        """
        Re-alignment via word_ids during inference:
        Records the model's prediction for the first subword of each word,
        discards predictions for subsequent fragments.
        Guarantees: If input has N words, output has exactly N BIO tags.
        """
        word_tags = []
        seen_words = set()

        for w_id, pred in zip(word_ids, subword_predictions):
            if w_id is None:
                continue
            if w_id not in seen_words:
                seen_words.add(w_id)
                word_tags.append(pred)

        # Safety fill if any trailing words were missing
        while len(word_tags) < len(words):
            word_tags.append("O")

        return word_tags
