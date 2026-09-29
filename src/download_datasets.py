"""
BioSpan AI - Real Dataset Fetcher & Parser
Downloads and parses the gold-standard NCBI Disease Corpus
(Hugging Face: `ncbi/ncbi_disease` / PubMed / Google Scholar citation: Doğan et al., JBI 2014)
"""

import os
import urllib.request
import csv
import json

BASE_URL = "https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def download_file(filename: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    target_path = os.path.join(DATA_DIR, filename)
    url = BASE_URL + filename
    print(f"Downloading authentic NCBI Disease dataset from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as response, open(target_path, "wb") as out_file:
        out_file.write(response.read())
    print(f"Saved to {target_path} (Size: {os.path.getsize(target_path)} bytes)")
    return target_path

def parse_conll(filepath: str):
    sentences = []
    current_tokens = []
    current_tags = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append((current_tokens, current_tags))
                    current_tokens = []
                    current_tags = []
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_tags.append(parts[1])

    if current_tokens:
        sentences.append((current_tokens, current_tags))
    return sentences

def extract_entities_from_tags(tokens, tags):
    entities = []
    current_ent = None
    for idx, (t, tag) in enumerate(zip(tokens, tags)):
        if tag == "B-Disease":
            if current_ent:
                entities.append(current_ent)
            current_ent = {"text": t, "start": idx, "end": idx, "tokens": [t], "tags": [tag]}
        elif tag == "I-Disease":
            if current_ent:
                current_ent["text"] += " " + t
                current_ent["end"] = idx
                current_ent["tokens"].append(t)
                current_ent["tags"].append(tag)
        else:
            if current_ent:
                entities.append(current_ent)
                current_ent = None
    if current_ent:
        entities.append(current_ent)
    return entities

def build_datasets():
    # 1. Download test.tsv and devel.tsv
    test_tsv = download_file("test.tsv")
    devel_tsv = download_file("devel.tsv")

    # 2. Parse sentences
    test_sents = parse_conll(test_tsv)
    print(f"Parsed {len(test_sents)} real test sentences from NCBI Disease corpus.")

    # 3. Create real test CSV for Batch Pipeline (sentence_id, text)
    csv_path = os.path.join(DATA_DIR, "ncbi_disease_test.csv")
    full_csv_path = os.path.join(DATA_DIR, "ncbi_disease_gold.csv")
    json_cases_path = os.path.join(DATA_DIR, "ncbi_disease_cases.json")

    case_studies = []
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f_test, \
         open(full_csv_path, "w", newline="", encoding="utf-8") as f_gold:
        
        w_test = csv.writer(f_test)
        w_gold = csv.writer(f_gold)
        
        w_test.writerow(["sentence_id", "text"])
        w_gold.writerow(["sentence_id", "text", "word_count", "gold_tags", "entity_count", "entities"])

        for i, (tokens, tags) in enumerate(test_sents):
            sent_id = f"NCBI_TEST_{i+1:04d}"
            # Reconstruct natural text (handling punctuation spacing)
            text = " ".join(tokens)
            tag_str = " ".join(tags)
            ents = extract_entities_from_tags(tokens, tags)
            ent_names = [e["text"] for e in ents]

            w_test.writerow([sent_id, text])
            w_gold.writerow([sent_id, text, len(tokens), tag_str, len(ents), " | ".join(ent_names)])

            # Select high-impact multi-word sentences for the curated case studies
            if len(ents) >= 2 and any(len(e["tokens"]) >= 2 for e in ents) and len(case_studies) < 12:
                case_studies.append({
                    "id": sent_id,
                    "title": f"NCBI PubMed Benchmark: {ents[0]['text'].title()}",
                    "source": "NCBI Disease Corpus (Hugging Face / PubMed / Google Scholar)",
                    "text": text,
                    "tokens": tokens,
                    "gold_tags": tags,
                    "gold_tags_str": tag_str,
                    "expected_entities": ent_names,
                    "clinical_importance": f"Authentic peer-reviewed abstract from NCBI Disease benchmark. Demonstrates complex boundary detection: {', '.join(ent_names)}."
                })

    with open(json_cases_path, "w", encoding="utf-8") as f_json:
        json.dump(case_studies, f_json, indent=2)

    print(f"Generated {csv_path} with {len(test_sents)} sentences.")
    print(f"Generated {full_csv_path} with gold annotations.")
    print(f"Generated {json_cases_path} with {len(case_studies)} curated NCBI PubMed case studies.")

if __name__ == "__main__":
    build_datasets()
