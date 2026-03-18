# Document Intelligence App

A **Streamlit** application that extracts structured data from PDF or scanned images (OCR + layout understanding), detects entities, and enables searching for similar documents.

## Features

- OCR using **EasyOCR** (with optional Tesseract fallback)
- Layout-aware extraction (tables / text blocks)
- Named Entity Recognition (spaCy)
- Semantic search over previously processed papers (sentence-transformers)
- Export extraction as clean JSON
- Batch-ingest a folder of PDFs/images for bulk processing (100+ docs)
- Per-file ingest logging (success/failure) + dedupe by file hash
- Preview extracted tables during batch ingest
- Better table extraction via Donut/parsed layout tables when available
- Handwriting OCR support via TrOCR (Microsoft)
- Document summarization (3-sentence summary) after extraction
- Export combined Excel report (one row per document)
- Document comparison / change detection (highlight diffs between two documents)

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **System requirements:**
> - **Tesseract** must be installed on your machine for full OCR support.
>   - **Windows:** https://github.com/tesseract-ocr/tesseract/wiki
>   - **macOS:** `brew install tesseract`
> - **Poppler** is required for PDF rendering via `pdf2image`.
>   - **Windows:** download from https://github.com/oschwartz10612/poppler-windows/releases
>   - **macOS:** `brew install poppler`

### 2) (Optional) Install spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 3) Run the app locally

```bash
streamlit run app.py
```

### 4) Deploy to Streamlit Cloud (fastest)

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and connect your GitHub account.
3. Create a new app and point it to this repo + the `app.py` file.

Streamlit Cloud will automatically install dependencies from `requirements.txt` and run `streamlit run app.py`.

> Tip: If you want free hosting, choose the free plan.

### 5) Run from Google Colab (quick test)

1. Open the notebook `streamlit_colab.ipynb` in Google Colab.
2. Replace `<YOUR_USER>` and `<YOUR_REPO>` in the first cell to match your GitHub repo.
3. Run all cells.

> The notebook runs Streamlit in the background and prints a public URL (via ngrok) to open in your browser.

### 6) (Optional) Run CLI ingest / repair

```bash
python cli.py --ingest-folder /path/to/papers --skip-duplicates
python cli.py --list
python cli.py --repair 123
```

> Tip: Use `--use-trocr` for handwritten pages and `--use-donut` for layout-aware parsing.

## How it works

1. Upload a PDF or image file.
2. The app runs OCR to extract text blocks and bounding boxes.
3. It uses a lightweight layout parser to detect tables and structured blocks.
4. It runs NER on the extracted text (organizations, persons, dates, etc.).
5. It stores each ingested document in a local SQLite database along with an embedding.
6. When you upload a new document, it finds the most relevant existing papers using semantic similarity.

## Project structure

- `app.py` - Streamlit frontend
- `docintelligence/` - core extraction and search modules
- `data/` - storage for the SQLite database and uploads

---

## Notes

- This implementation is intentionally lightweight and can be extended with advanced layout models like LayoutLM/Donut.
- For best search quality, process a set of reference papers (PDFs) in advance and store them in the database.
