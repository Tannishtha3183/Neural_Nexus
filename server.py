"""
BioSpan AI - Production FastAPI Server
Serves the REST API Layer and the Interactive Biomedical NER Dashboard.
"""

import os
import sys
import io
import csv
import json
import re
import urllib.request
import urllib.parse
from typing import List, Optional, Dict, Any

import pypdf
import docx
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.model import BioSpanEngine
from src.biosanitizer import BioSanitizer
from src.alignment import SubwordAligner
from src.ontology import ICD10Linker
from src.metrics import MetricEvaluator

# Initialize Core Services
app = FastAPI(
    title="BioSpan AI - Biomedical NER & Clinical Ontology Engine",
    description="Domain-Specific BioBERT Representation, First-Subword Alignment, and Deterministic BIO-Sanitizer",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = BioSpanEngine()
sanitizer = BioSanitizer(target_entity="Disease")
aligner = SubwordAligner()
ontology_linker = ICD10Linker()
evaluator = MetricEvaluator(target_entity="Disease")

# Request Models
class PredictRequest(BaseModel):
    text: str
    simulate_orphaned_i_prob: Optional[float] = 0.0

class SubwordRequest(BaseModel):
    text: str
    gold_tags: Optional[List[str]] = None

class SanitizeDebugRequest(BaseModel):
    words: List[str]
    raw_tags: List[str]

class MetricsEvalRequest(BaseModel):
    words: List[str]
    gold_tags: List[str]
    predicted_tags: List[str]
    sanitized_tags: Optional[List[str]] = None

# API Endpoints
@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "system": "BioSpan AI",
        "version": "1.0.0",
        "model_backbone": engine.model_name,
        "device": str(engine.device),
        "subword_loss_masking": "ignore_index=-100",
        "bio_sanitizer_active": True,
        "icd10_ontology_size": len(ontology_linker.get_all_codes())
    }

@app.get("/api/sample-cases")
def get_sample_cases():
    # Prefer authentic NCBI Disease PubMed benchmark cases from Hugging Face / Google Scholar
    ncbi_path = os.path.join(os.path.dirname(__file__), "data", "ncbi_disease_cases.json")
    if os.path.exists(ncbi_path):
        with open(ncbi_path, "r", encoding="utf-8") as f:
            return json.load(f)
    cases_path = os.path.join(os.path.dirname(__file__), "data", "sample_cases.json")
    if os.path.exists(cases_path):
        with open(cases_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.get("/api/dataset/ncbi-cases")
def get_ncbi_cases():
    ncbi_path = os.path.join(os.path.dirname(__file__), "data", "ncbi_disease_cases.json")
    if os.path.exists(ncbi_path):
        with open(ncbi_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@app.get("/api/dataset/ncbi-test-batch")
def get_ncbi_test_batch(limit: int = 15):
    """Returns authentic test sentences from NCBI Disease Corpus (Hugging Face / PubMed)."""
    csv_path = os.path.join(os.path.dirname(__file__), "data", "ncbi_disease_test.csv")
    items = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                if idx >= limit:
                    break
                items.append({
                    "sentence_id": row["sentence_id"],
                    "text": row["text"]
                })
    return items

@app.get("/api/dataset/download-ncbi-csv")
def download_ncbi_csv():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "ncbi_disease_test.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="ncbi_disease_test_corpus.csv")
    raise HTTPException(status_code=404, detail="NCBI Disease test dataset not found.")

@app.post("/api/predict")
def predict_endpoint(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Clinical text cannot be empty.")
    
    result = engine.predict(req.text, simulate_orphaned_i_prob=req.simulate_orphaned_i_prob or 0.0)
    return result

@app.post("/api/subword-align")
def subword_align_endpoint(req: SubwordRequest):
    words = engine.tokenize_text(req.text)
    if not words:
        raise HTTPException(status_code=400, detail="Text must contain at least one valid word.")
    
    gold_tags = req.gold_tags
    if not gold_tags or len(gold_tags) != len(words):
        # Infer default tags from engine
        pred_res = engine.predict(req.text)
        gold_tags = pred_res["sanitized_tags"]

    alignment_res = aligner.align_sequence(words, gold_tags)
    return alignment_res

@app.post("/api/sanitize-debug")
def sanitize_debug_endpoint(req: SanitizeDebugRequest):
    if len(req.words) != len(req.raw_tags):
        raise HTTPException(status_code=400, detail="Word count must match raw tag count.")
    
    result = sanitizer.sanitize(req.words, req.raw_tags)
    return result

@app.post("/api/metrics-eval")
def metrics_eval_endpoint(req: MetricsEvalRequest):
    if len(req.words) != len(req.gold_tags) or len(req.words) != len(req.predicted_tags):
        raise HTTPException(status_code=400, detail="Words, gold tags, and predicted tags must have equal lengths.")
    
    # Auto-sanitize if not provided
    sanitized_tags = req.sanitized_tags
    if not sanitized_tags:
        san_res = sanitizer.sanitize(req.words, req.predicted_tags)
        sanitized_tags = san_res["sanitized_tags"]

    comparison = evaluator.run_comprehensive_comparison(
        req.words, req.gold_tags, req.predicted_tags, sanitized_tags
    )
    return comparison

@app.get("/api/ontology")
def ontology_search(q: Optional[str] = None, category: Optional[str] = None):
    results = ontology_linker.search_codes(query=q, category=category)
    return {
        "status": "success",
        "total": len(results),
        "matches": results,
        "results": results
    }


@app.get("/api/system/diagnostics")
def system_diagnostics():
    import time
    start_time = time.time()
    test_note = "Patient diagnosed with stage IV non-small cell lung cancer and cough. No history of myocardial infarction."
    test_res = engine.predict(test_note)
    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "status": "ONLINE",
        "latency_ms": latency_ms,
        "test_entities_extracted": test_res["entity_count"],
        "device": str(engine.device).upper(),
        "model_checkpoint": engine.model_name,
        "hidden_dim": engine.hidden_dim,
        "attention_heads": engine.attention_heads,
        "transformer_layers": engine.transformer_layers,
        "parameters": "108M Parameters",
        "training_corpus": "NCBI Disease Corpus (PubMed / PMC)",
        "subword_loss_masking": "ignore_index=-100",
        "bio_sanitizer": "Deterministic State Machine Active",
        "ontology_size": len(ontology_linker.get_all_codes())
    }


def extract_text_from_file(filename: str, content: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    
    if ext == ".pdf":
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            extracted_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(page_text.strip())
            text = "\n\n".join(extracted_pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")
            
    elif ext in [".docx", ".doc"]:
        try:
            doc = docx.Document(io.BytesIO(content))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            text = "\n".join(paragraphs)
        except Exception as e:
            try:
                raw = content.decode("utf-8", errors="replace")
                text = "".join(c for c in raw if c.isprintable() or c in "\n\r\t ")
            except Exception:
                raise HTTPException(status_code=400, detail=f"Failed to read Word document: {str(e)}")
                
    elif ext in [".txt", ".md", ".rtf", ".log", ".csv"]:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="replace")
            
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or contains no readable clinical text.")
        
    return text


@app.post("/api/upload-document")
async def upload_document_endpoint(file: UploadFile = File(...)):
    """
    Accepts clinical report files (PDF, Word .docx/.doc, TXT),
    extracts the text, and immediately runs BioBERT inference and ICD-10 linking.
    """
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    filename = file.filename or "uploaded_report"
    ext = os.path.splitext(filename)[1].lower()
    
    extracted_text = extract_text_from_file(filename, content)
    
    # Run immediate BioBERT prediction
    prediction = engine.predict(extracted_text)
    
    return {
        "status": "success",
        "filename": filename,
        "file_type": ext,
        "char_count": len(extracted_text),
        "word_count": len(extracted_text.split()),
        "text": extracted_text,
        "extracted_text": extracted_text,
        "entities": prediction.get("entities", []),
        "prediction": prediction
    }


class TranslateRequest(BaseModel):
    text: str
    target_lang: str


@app.post("/api/translate")
def translate_text_endpoint(req: TranslateRequest):
    """
    Translates text into any Indian language (hi, bn, te, ta, mr, gu, kn, ml, pa, or, as, ur).
    Uses lightweight translation proxy with caching.
    """
    target = req.target_lang.lower().strip()
    text = req.text.strip()
    if not text or target in ["en", "english"]:
        return {"translated_text": text, "target_lang": target}
        
    indian_langs = {"hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"}
    if target not in indian_langs:
        return {"translated_text": text, "target_lang": target}
        
    try:
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target}&dt=t&q={urllib.parse.quote(text)}"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(request, timeout=4) as response:
            data = json.loads(response.read().decode("utf-8"))
            translated = "".join(item[0] for item in data[0] if item[0])
            return {"translated_text": translated, "target_lang": target}
    except Exception as e:
        return {"translated_text": text, "target_lang": target, "error": str(e)}


@app.post("/api/batch")
async def batch_process_endpoint(file: UploadFile = File(...)):
    """
    Processes test.csv containing 'sentence_id' and 'text'.
    Generates submission.csv with space-delimited BIO tags matching original word count.
    """
    content = await file.read()
    decoded = content.decode("utf-8", errors="replace")
    
    csv_reader = csv.DictReader(io.StringIO(decoded))
    fieldnames = csv_reader.fieldnames or []
    
    # Identify ID column and text column
    id_col = None
    text_col = None
    for f in fieldnames:
        f_lower = f.lower().strip()
        if "id" in f_lower:
            id_col = f
        if "text" in f_lower or "sentence" in f_lower or "words" in f_lower:
            text_col = f
            
    if not id_col or not text_col:
        # Fallback: assume first column is ID, second is text
        if len(fieldnames) >= 2:
            id_col, text_col = fieldnames[0], fieldnames[1]
        else:
            raise HTTPException(status_code=400, detail="CSV must contain at least two columns: sentence_id and text/sentence.")

    results = []
    submission_rows = []
    total_violations_repaired = 0
    total_spans_extracted = 0

    for row in csv_reader:
        sent_id = row.get(id_col, "").strip()
        raw_text = row.get(text_col, "").strip()
        if not sent_id or not raw_text:
            continue
        
        pred = engine.predict(raw_text)
        tag_str = pred["sanitized_submission_string"]
        total_violations_repaired += pred["sanitizer_report"]["violation_count"]
        total_spans_extracted += pred["entity_count"]

        submission_rows.append({
            "sentence_id": sent_id,
            "tags": tag_str
        })

        results.append({
            "sentence_id": sent_id,
            "text": raw_text,
            "word_count": pred["token_count"],
            "tags": tag_str,
            "entities": pred["entities"],
            "violations_repaired": pred["sanitizer_report"]["violation_count"]
        })

    # Generate CSV string
    output_buffer = io.StringIO()
    writer = csv.DictWriter(output_buffer, fieldnames=["sentence_id", "tags"])
    writer.writeheader()
    writer.writerows(submission_rows)
    csv_data = output_buffer.getvalue()

class BatchJsonRequest(BaseModel):
    items: List[Dict[str, str]] # [{"sentence_id": "...", "text": "..."}]

@app.post("/api/batch-json")
def batch_json_endpoint(req: BatchJsonRequest):
    results = []
    submission_rows = []
    total_violations_repaired = 0
    total_spans_extracted = 0

    for item in req.items:
        sent_id = item.get("sentence_id", "").strip()
        raw_text = item.get("text", "").strip()
        if not sent_id or not raw_text:
            continue
        
        pred = engine.predict(raw_text)
        tag_str = pred["sanitized_submission_string"]
        total_violations_repaired += pred["sanitizer_report"]["violation_count"]
        total_spans_extracted += pred["entity_count"]

        submission_rows.append({
            "sentence_id": sent_id,
            "tags": tag_str
        })

        results.append({
            "sentence_id": sent_id,
            "text": raw_text,
            "word_count": pred["token_count"],
            "tags": tag_str,
            "entities": pred["entities"],
            "violations_repaired": pred["sanitizer_report"]["violation_count"]
        })

    output_buffer = io.StringIO()
    writer = csv.DictWriter(output_buffer, fieldnames=["sentence_id", "tags"])
    writer.writeheader()
    writer.writerows(submission_rows)
    csv_data = output_buffer.getvalue()

    return {
        "status": "success",
        "processed_sentences": len(results),
        "total_entities_extracted": total_spans_extracted,
        "total_violations_repaired": total_violations_repaired,
        "results_preview": results,
        "submission_csv_content": csv_data,
        "csv_download_ready": True
    }

# Mount Web Directory
web_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=os.path.join(web_dir)), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(os.path.dirname(__file__), "web", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>BioSpan AI API is Running.</h1><p>Visit /docs for API documentation.</p>")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
