"""
BioSpan AI - Evaluation & Defense Masterclass Engine
Implements strict mathematical comparison across:
1. Token-Level Accuracy (The Flawed Baseline)
2. Token-Level F1 (The Misleading Partial Credit Metric)
3. Exact Entity Span F1 (The Clinical Gold Standard)
"""

from typing import List, Dict, Tuple, Any, Set

class MetricEvaluator:
    """
    Evaluation Engine comparing Token Accuracy vs Token F1 vs Exact Span F1.
    """

    def __init__(self, target_entity: str = "Disease"):
        self.target_entity = target_entity
        self.b_tag = f"B-{target_entity}"
        self.i_tag = f"I-{target_entity}"
        self.o_tag = "O"

    def evaluate_exact_spans(self, gold_tags: List[str], pred_tags: List[str]) -> Dict[str, Any]:
        """
        Exact Entity Span F1 Calculation:
        An entity is correct IF AND ONLY IF the start boundary, end boundary,
        and entity label match the gold reference annotation precisely.
        """
        gold_spans = self._extract_span_tuples(gold_tags)
        pred_spans = self._extract_span_tuples(pred_tags)

        gold_set: Set[Tuple[int, int, str]] = set(gold_spans)
        pred_set: Set[Tuple[int, int, str]] = set(pred_spans)

        true_positives = len(gold_set.intersection(pred_set))
        false_positives = len(pred_set - gold_set)
        false_negatives = len(gold_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "metric_name": "Exact Entity Span F1 (Clinical Gold Standard)",
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "gold_spans_count": len(gold_spans),
            "pred_spans_count": len(pred_spans),
            "gold_spans": [f"[{s}:{e}] {t}" for s, e, t in gold_spans],
            "pred_spans": [f"[{s}:{e}] {t}" for s, e, t in pred_spans],
            "strict_pass": (false_positives == 0 and false_negatives == 0 and len(gold_spans) > 0)
        }

    def evaluate_token_f1(self, gold_tags: List[str], pred_tags: List[str]) -> Dict[str, Any]:
        """
        Token-Level F1: Evaluates each individual token's entity status.
        Rewards partial span overlap (e.g. 2 out of 5 words = 40% partial credit).
        """
        tp = 0
        fp = 0
        fn = 0

        for g, p in zip(gold_tags, pred_tags):
            g_is_ent = (g != self.o_tag)
            p_is_ent = (p != self.o_tag)

            if g_is_ent and p_is_ent:
                if g == p:
                    tp += 1
                else:
                    fp += 1
            elif not g_is_ent and p_is_ent:
                fp += 1
            elif g_is_ent and not p_is_ent:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "metric_name": "Token-Level F1 (Partial Credit Flawed)",
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "clinical_flaw": "Gives deceptive partial credit when disease phenotype modifiers are omitted."
        }

    def evaluate_token_accuracy(self, gold_tags: List[str], pred_tags: List[str]) -> Dict[str, Any]:
        """
        Token Accuracy: Total correct tokens / Total tokens.
        Highly misleading in NER due to class imbalance (~90% O tokens).
        """
        total = len(gold_tags)
        correct = sum(1 for g, p in zip(gold_tags, pred_tags) if g == p)
        accuracy = correct / total if total > 0 else 0.0

        # Trivial all-O baseline
        trivial_o_correct = sum(1 for g in gold_tags if g == self.o_tag)
        trivial_o_accuracy = trivial_o_correct / total if total > 0 else 0.0

        return {
            "metric_name": "Token Accuracy (Heavily Distorted)",
            "accuracy": round(accuracy * 100, 2),
            "correct_tokens": correct,
            "total_tokens": total,
            "trivial_o_baseline": round(trivial_o_accuracy * 100, 2),
            "clinical_flaw": f"Guessing 'O' everywhere yields a fake {round(trivial_o_accuracy * 100, 1)}% accuracy while finding zero diseases."
        }

    def run_comprehensive_comparison(self, words: List[str], gold_tags: List[str], 
                                    raw_pred_tags: List[str], 
                                    sanitized_pred_tags: List[str] = None) -> Dict[str, Any]:
        """
        Runs complete 3-metric defense benchmark comparing:
        1. Token Accuracy
        2. Token F1
        3. Exact Span F1 (Raw Model vs BIO-Sanitized Model)
        """
        raw_exact = self.evaluate_exact_spans(gold_tags, raw_pred_tags)
        raw_token_f1 = self.evaluate_token_f1(gold_tags, raw_pred_tags)
        raw_acc = self.evaluate_token_accuracy(gold_tags, raw_pred_tags)

        sanitized_exact = None
        sanitized_delta = None

        if sanitized_pred_tags:
            sanitized_exact = self.evaluate_exact_spans(gold_tags, sanitized_pred_tags)
            delta_f1 = round(sanitized_exact["f1"] - raw_exact["f1"], 2)
            sanitized_delta = {
                "f1_delta": f"{'+' if delta_f1 >= 0 else ''}{delta_f1}%",
                "rescued_spans": sanitized_exact["true_positives"] - raw_exact["true_positives"],
                "eliminated_hallucinations": raw_exact["false_positives"] - sanitized_exact["false_positives"]
            }

        return {
            "token_count": len(words),
            "words": words,
            "gold_tags": gold_tags,
            "raw_predictions": raw_pred_tags,
            "sanitized_predictions": sanitized_pred_tags,
            "token_accuracy": raw_acc,
            "token_f1": raw_token_f1,
            "exact_span_f1_raw": raw_exact,
            "exact_span_f1_sanitized": sanitized_exact,
            "sanitizer_delta": sanitized_delta,
            "defense_verdict": self._generate_defense_verdict(raw_acc, raw_token_f1, raw_exact, sanitized_exact)
        }

    def _extract_span_tuples(self, tags: List[str]) -> List[Tuple[int, int, str]]:
        """Extracts (start, end, label) tuples strictly starting with B-."""
        spans = []
        current = None

        for idx, tag in enumerate(tags):
            if tag == self.b_tag:
                if current:
                    spans.append(current)
                current = (idx, idx, self.target_entity)
            elif tag == self.i_tag:
                if current:
                    current = (current[0], idx, self.target_entity)
            else:
                if current:
                    spans.append(current)
                    current = None

        if current:
            spans.append(current)
        return spans

    def _generate_defense_verdict(self, acc, tok_f1, raw_exact, san_exact) -> str:
        res = []
        res.append(f"1. Token Accuracy ({acc['accuracy']}%) is invalid for clinical evaluation due to the {acc['trivial_o_baseline']}% 'O'-class dominance.")
        res.append(f"2. Token F1 ({tok_f1['f1']}%) masks phenotype modifier truncations by awarding partial credit.")
        res.append(f"3. Exact Span F1 is {raw_exact['f1']}% raw.")
        if san_exact and san_exact['f1'] > raw_exact['f1']:
            res.append(f"4. The Deterministic BIO-Sanitizer successfully increased Exact Span F1 to {san_exact['f1']}% by repairing orphaned I-Disease boundaries.")
        return " ".join(res)
