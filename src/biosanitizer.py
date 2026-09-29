"""
BioSpan AI - Deterministic BIO-Sanitizer State Machine
Enforces grammatical rules of the BIO sequence labeling scheme.
Eliminates statistical hallucinations and resolves orphaned I-Disease tags.
"""

from typing import List, Dict, Tuple, Any

class BioSanitizer:
    """
    Deterministic Post-Processing State Machine for BIO Tagging.
    
    Rule Matrix:
    - O -> O           : Valid
    - O -> B-Disease   : Valid (Initiates a new disease entity)
    - O -> I-Disease   : ILLEGAL (Orphaned continuation). Rewired to B-Disease.
    - START -> I-Disease: ILLEGAL (Orphaned continuation). Rewired to B-Disease.
    - B-Disease -> I-Disease: Valid (Continues the current disease entity)
    - B-Disease -> B-Disease: Valid (Directly starts a subsequent adjacent disease entity)
    - B-Disease -> O   : Valid (Closes current disease entity)
    - I-Disease -> I-Disease: Valid (Continues current disease entity)
    - I-Disease -> B-Disease: Valid (Closes current disease entity, starts new adjacent entity)
    - I-Disease -> O   : Valid (Closes current disease entity)
    """

    def __init__(self, target_entity: str = "Disease"):
        self.target_entity = target_entity
        self.b_tag = f"B-{target_entity}"
        self.i_tag = f"I-{target_entity}"
        self.o_tag = "O"

    def sanitize(self, words: List[str], raw_tags: List[str], confidences: List[float] = None) -> Dict[str, Any]:
        """
        Parses raw model predictions, identifies grammatical violations,
        and rewires illegal transitions into legal BIO sequences.
        """
        sanitized_tags = list(raw_tags)
        violations = []
        repairs = []
        
        prev_tag = self.o_tag
        
        for idx, (word, tag) in enumerate(zip(words, raw_tags)):
            is_orphaned_i = False
            
            if tag == self.i_tag:
                # Violation Case 1: Orphaned I-Disease at sequence start
                if idx == 0:
                    is_orphaned_i = True
                    reason = "Orphaned I-Disease tag at start of sequence (no preceding B-Disease)."
                # Violation Case 2: Orphaned I-Disease preceded by O
                elif prev_tag == self.o_tag:
                    is_orphaned_i = True
                    reason = f"Orphaned I-Disease tag preceded by 'O' token ('{words[idx-1]}')."
            
            if is_orphaned_i:
                sanitized_tags[idx] = self.b_tag
                violation_record = {
                    "index": idx,
                    "word": word,
                    "original_tag": tag,
                    "sanitized_tag": self.b_tag,
                    "reason": reason,
                    "previous_word": words[idx-1] if idx > 0 else "<START>",
                    "previous_tag": prev_tag if idx > 0 else "<START>",
                    "action_taken": f"Dynamically rewired '{tag}' -> '{self.b_tag}' to restore grammatical legality.",
                    "clinical_impact": "Rescued entity boundary. Prevents False Negative (missing span) and False Positive (hallucinated partial span)."
                }
                violations.append(violation_record)
                repairs.append(violation_record)
                prev_tag = self.b_tag
            else:
                prev_tag = tag

        # Extract entities from both raw and sanitized tags for comparison
        raw_spans = self.extract_spans(words, raw_tags, confidences, allow_orphaned=False)
        sanitized_spans = self.extract_spans(words, sanitized_tags, confidences, allow_orphaned=False)
        
        return {
            "words": words,
            "raw_tags": raw_tags,
            "sanitized_tags": sanitized_tags,
            "has_violations": len(violations) > 0,
            "violation_count": len(violations),
            "repairs": repairs,
            "raw_spans": raw_spans,
            "sanitized_spans": sanitized_spans,
            "span_recovery_count": len(sanitized_spans) - len(raw_spans)
        }

    def extract_spans(self, words: List[str], tags: List[str], 
                      confidences: List[float] = None, 
                      allow_orphaned: bool = False) -> List[Dict[str, Any]]:
        """
        Extracts multi-word entity spans with exact word indices and average confidences.
        Strict standard: An entity starts ONLY at B-Disease, continues through I-Disease,
        and terminates at O or B-Disease or end of sentence.
        """
        spans = []
        current_span = None
        
        for idx, (word, tag) in enumerate(zip(words, tags)):
            conf = confidences[idx] if confidences and idx < len(confidences) else 1.0
            
            if tag == self.b_tag:
                # If there's an open span, close it and record it
                if current_span:
                    spans.append(self._finalize_span(current_span))
                # Start new span
                current_span = {
                    "start_token": idx,
                    "end_token": idx,
                    "words": [word],
                    "tags": [tag],
                    "confidences": [conf],
                    "entity_type": self.target_entity
                }
            elif tag == self.i_tag:
                if current_span:
                    # Legal continuation
                    current_span["end_token"] = idx
                    current_span["words"].append(word)
                    current_span["tags"].append(tag)
                    current_span["confidences"].append(conf)
                else:
                    # Orphaned I-tag encountered
                    if allow_orphaned:
                        current_span = {
                            "start_token": idx,
                            "end_token": idx,
                            "words": [word],
                            "tags": [tag],
                            "confidences": [conf],
                            "entity_type": self.target_entity,
                            "is_orphaned": True
                        }
                    else:
                        # In strict evaluation, orphaned I tags with no preceding B are discarded as broken boundaries
                        pass
            else:
                # O tag closes any active span
                if current_span:
                    spans.append(self._finalize_span(current_span))
                    current_span = None

        if current_span:
            spans.append(self._finalize_span(current_span))
            
        return spans

    def _finalize_span(self, span_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates composite properties for an extracted entity span."""
        confs = span_dict["confidences"]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        min_conf = min(confs) if confs else 0.0
        entity_text = " ".join(span_dict["words"])
        
        return {
            "entity_text": entity_text,
            "entity_type": span_dict["entity_type"],
            "start_token": span_dict["start_token"],
            "end_token": span_dict["end_token"],
            "token_count": len(span_dict["words"]),
            "tokens": span_dict["words"],
            "tags": span_dict["tags"],
            "mean_confidence": round(mean_conf, 4),
            "min_confidence": round(min_conf, 4),
            "confidences": [round(c, 4) for c in confs],
            "is_orphaned": span_dict.get("is_orphaned", False)
        }
