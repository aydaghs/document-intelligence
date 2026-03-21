import difflib
import io
import json
import os
import re
import traceback
from typing import Callable, Optional

import pandas as pd
import streamlit as st

IMPORT_ERROR = None
try:
    from docintelligence import (
        DocumentStorage,
        SemanticSearch,
        extract_entities,
        extract_key_phrases,
        ocr_image,
        ocr_pdf,
        parse_layout,
        summarize_text,
        file_hash,
        ensure_dir,
    )

    try:
        from docintelligence import extract_with_donut
    except ImportError:
        extract_with_donut = None

    try:
        from docintelligence import trocr_ocr
    except ImportError:
        trocr_ocr = None

except Exception as e:
    IMPORT_ERROR = e
    DocumentStorage = None          # type: ignore
    SemanticSearch = None           # type: ignore
    extract_entities = None         # type: ignore
    extract_key_phrases = None      # type: ignore
    ocr_image = None                # type: ignore
    ocr_pdf = None                  # type: ignore
    parse_layout = None             # type: ignore
    summarize_text = None           # type: ignore
    file_hash = None                # type: ignore
    ensure_dir = None               # type: ignore
    extract_with_donut = None       # type: ignore
    trocr_ocr = None                # type: ignore


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "documents.db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_uploaded_file(uploaded_file) -> str:
    ensure_dir(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


@st.cache_resource
def get_storage() -> "DocumentStorage":
    ensure_dir(DATA_DIR)
    return DocumentStorage(DB_PATH)


@st.cache_resource
def get_search() -> Optional["SemanticSearch"]:
    try:
        s = SemanticSearch()
        return s
    except Exception as exc:
        return None


def _highlight_query(text: str, query: str) -> str:
    if not query or not text:
        return text

    def repl(match):
        return f"**{match.group(0)}**"

    return re.compile(re.escape(query), re.IGNORECASE).sub(repl, text)


def _diff_text_md(a: str, b: str) -> str:
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    diff = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))
    if not diff:
        return "No differences detected."
    return "\n".join(["```diff"] + diff + ["```"])


def _render_import_error() -> None:
    st.error("Import error — a required component failed to load.")
    if IMPORT_ERROR is not None:
        st.code(traceback.format_exception_only(type(IMPORT_ERROR), IMPORT_ERROR)[-1].strip())
        with st.expander("Full traceback"):
            st.code("\n".join(traceback.format_exception(
                type(IMPORT_ERROR), IMPORT_ERROR, IMPORT_ERROR.__traceback__
            )))


def _require_imports() -> bool:
    if IMPORT_ERROR is not None:
        _render_import_error()
        return False
    return True


def _build_query_text(extracted: dict) -> str:
    parts = [extracted.get("text", "")]
    for t in extracted.get("tables", []):
        for row in t.get("rows", []):
            parts.append(" ".join(str(c) for c in row))
    return "\n".join(parts)


def _process_and_store(
    file_path: str,
    filename: str,
    storage: "DocumentStorage",
    search: Optional["SemanticSearch"],
    use_donut: bool,
    use_trocr: bool,
    use_summary: bool,
    skip_duplicates: bool,
    use_semantic_search: bool,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    title: Optional[str] = None,
) -> dict:
    """Process a file (OCR + layout + NLP + embeddings) and save to storage."""

    from PIL import Image

    file_hash_val = file_hash(file_path)
    if skip_duplicates and storage.has_document_hash(file_hash_val):
        return {"status": "skipped", "reason": "already ingested", "file_hash": file_hash_val}

    is_pdf = file_path.lower().endswith(".pdf")
    use_pdfplumber_fallback = False
    pdfplumber_tables = []
    pages = []

    if is_pdf:
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(file_path, dpi=300)
        except Exception:
            import pdfplumber
            use_pdfplumber_fallback = True
            with pdfplumber.open(file_path) as pdf:
                for p in pdf.pages:
                    pages.append({"text": p.extract_text() or "", "page_number": p.page_number})
                    try:
                        tables = p.extract_tables()
                        if tables:
                            pdfplumber_tables.extend(tables)
                    except Exception:
                        pass
    else:
        pages = [Image.open(file_path).convert("RGB")]

    donut_result = None
    if use_donut and pages and extract_with_donut is not None and not use_pdfplumber_fallback:
        try:
            donut_result = extract_with_donut(pages[0])
        except Exception:
            donut_result = None

    ocr_results = []
    total_pages = len(pages)
    for idx, page in enumerate(pages, start=1):
        if progress_callback:
            progress_callback(idx, total_pages, f"Processing page {idx}/{total_pages}")

        if use_pdfplumber_fallback and isinstance(page, dict):
            ocr_results.append({
                "page": page.get("page_number", idx),
                "blocks": [{"text": page.get("text", ""), "confidence": None,
                             "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]]}],
            })
            continue

        if use_trocr and trocr_ocr is not None:
            lines = trocr_ocr(page)
            blocks = [{"text": l, "confidence": None, "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]]}
                      for l in lines]
        else:
            blocks = ocr_image(page)
        ocr_results.append({"page": idx, "blocks": blocks})

    layout = parse_layout(
        ocr_results,
        donut_parsed=donut_result.get("parsed") if donut_result else None,
    )

    if pdfplumber_tables:
        for t in pdfplumber_tables:
            layout["tables"].append({"source": "pdfplumber", "row_count": len(t), "rows": t})

    query_text = _build_query_text(layout)

    emb_blob = None
    if use_semantic_search and search is not None:
        try:
            emb = search.embed([query_text])[0]
            emb_blob = search.serialize_embedding(emb)
        except Exception:
            pass

    summary = None
    if use_summary:
        try:
            summary = summarize_text(query_text)
        except Exception:
            summary = None

    storage.add_document(
        filename=filename,
        file_hash=file_hash_val,
        title=title or filename,
        text=query_text,
        summary=summary,
        metadata={"source_path": file_path},
        embedding=emb_blob,
    )

    return {"status": "saved", "file_hash": file_hash_val, "layout": layout,
            "donut": donut_result, "summary": summary}


# ── main UI ──────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Document Intelligence", layout="wide", page_icon="📄")

    if not _require_imports():
        return

    st.title("📄 Document Intelligence")
    st.caption("OCR · Layout Analysis · Entity Extraction · Semantic Search")

    # ── sidebar ──
    st.sidebar.header("⚙️ Options")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
    )
    uploaded_files = st.sidebar.file_uploader(
        "Upload multiple files",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=True,
    )

    use_donut = st.sidebar.checkbox("Use Donut (layout-aware parsing)", value=False,
                                    help="Requires transformers + ~1GB model download")
    use_trocr = st.sidebar.checkbox("Use TrOCR (handwriting OCR)", value=False,
                                    help="Requires transformers + model download")
    use_summary = st.sidebar.checkbox("Generate summary", value=True)
    skip_duplicates = st.sidebar.checkbox("Skip already-ingested files", value=True)
    preview_tables = st.sidebar.checkbox("Preview tables during batch ingest", value=False)
    use_semantic_search = st.sidebar.checkbox("Enable semantic search", value=True)

    storage = get_storage()
    search = get_search()

    if use_semantic_search and search is None:
        st.sidebar.warning("Semantic search unavailable — install sentence-transformers.")
    elif use_semantic_search and search is not None:
        st.sidebar.info(f"Search backend: **{search.backend}**")

    # ── batch folder ingest ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Batch ingest folder")
    ingest_folder = st.sidebar.text_input("Folder path", value="")
    if st.sidebar.button("Ingest folder"):
        if not ingest_folder or not os.path.isdir(ingest_folder):
            st.sidebar.error("Provide a valid folder path.")
        else:
            exts = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp")
            file_paths = [
                os.path.join(root, fn)
                for root, _, files in os.walk(ingest_folder)
                for fn in files if fn.lower().endswith(exts)
            ]
            log_rows = []
            progress = st.sidebar.progress(0)
            for idx, fp in enumerate(file_paths):
                try:
                    result = _process_and_store(
                        fp, os.path.basename(fp), storage, search,
                        use_donut, use_trocr, use_summary, skip_duplicates,
                        use_semantic_search=use_semantic_search and search is not None,
                    )
                    status = result.get("status", "saved")
                    log_rows.append({"file": fp, "status": status,
                                     "summary": result.get("summary"), "reason": result.get("reason")})
                    if preview_tables and status == "saved":
                        st.sidebar.write(f"**{os.path.basename(fp)}**")
                        for t in (result.get("layout") or {}).get("tables", []):
                            st.sidebar.dataframe(t.get("rows", []))
                except Exception as e:
                    log_rows.append({"file": fp, "status": "error", "reason": str(e)})
                progress.progress((idx + 1) / max(1, len(file_paths)))

            st.sidebar.success(f"Ingested {len(file_paths)} file(s).")
            st.sidebar.dataframe(pd.DataFrame(log_rows))

    # ── multi-file upload ──
    if uploaded_files:
        st.subheader("📥 Batch upload results")
        file_log = []
        for uf in uploaded_files:
            fp = _save_uploaded_file(uf)
            pbar = st.progress(0)
            status_txt = st.empty()

            def _cb(cur, total, msg, _pbar=pbar, _st=status_txt):
                _pbar.progress(cur / max(1, total))
                _st.text(msg)

            try:
                result = _process_and_store(
                    fp, uf.name, storage, search,
                    use_donut, use_trocr, use_summary, skip_duplicates,
                    use_semantic_search=use_semantic_search and search is not None,
                    progress_callback=_cb,
                )
                file_log.append({"file": uf.name, "status": result.get("status"),
                                  "summary": result.get("summary"), "reason": result.get("reason")})
            except Exception as e:
                file_log.append({"file": uf.name, "status": "error", "reason": str(e)})

        st.dataframe(pd.DataFrame(file_log))

    # ── single file processing ──
    if uploaded_file is None:
        st.info("👈 Upload a PDF or scanned image in the sidebar to get started.")

        # Show feature status
        with st.expander("ℹ️ Feature availability"):
            try:
                import easyocr
                st.success("EasyOCR — installed (image OCR)")
            except ImportError:
                st.warning("EasyOCR — not installed (install `easyocr` for image/scanned-PDF OCR)")
            try:
                import pytesseract
                st.success("pytesseract — installed (image OCR fallback)")
            except ImportError:
                st.warning("pytesseract — not installed")
            try:
                import pdfplumber
                st.success("pdfplumber — installed (PDF text extraction ✓)")
            except ImportError:
                st.error("pdfplumber — not installed")
            try:
                import spacy
                st.success("spaCy — installed (NER)")
            except ImportError:
                st.warning("spaCy — not installed (regex NER fallback active)")
            try:
                import sentence_transformers
                st.success("sentence-transformers — installed (semantic search)")
            except ImportError:
                st.warning("sentence-transformers — not installed (TF-IDF search fallback active)")
            try:
                import transformers
                st.success("transformers — installed (BART summarization, Donut, TrOCR)")
            except ImportError:
                st.warning("transformers — not installed (extractive summarization fallback active)")
        return

    # ── process the uploaded file ──
    file_path = _save_uploaded_file(uploaded_file)
    st.success(f"Uploaded: **{uploaded_file.name}**")

    is_pdf = file_path.lower().endswith(".pdf")

    # PDF preview
    if is_pdf:
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                if pdf.pages:
                    pil_img = pdf.pages[0].to_image(resolution=150).original
                    st.image(pil_img, caption="PDF preview (page 1)", use_column_width=True)
        except Exception:
            pass
    else:
        try:
            st.image(file_path, caption="Uploaded image", use_column_width=True)
        except Exception:
            pass

    # Run OCR
    with st.spinner("Extracting text…"):
        use_pdfplumber_fallback = False
        pdfplumber_tables = []
        pages = []

        if is_pdf:
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(file_path, dpi=300)
            except Exception:
                import pdfplumber as _plumber
                use_pdfplumber_fallback = True
                with _plumber.open(file_path) as pdf:
                    for p in pdf.pages:
                        pages.append({"text": p.extract_text() or "", "page_number": p.page_number})
                        try:
                            tbls = p.extract_tables()
                            if tbls:
                                pdfplumber_tables.extend(tbls)
                        except Exception:
                            pass
        else:
            from PIL import Image as _Image
            pages = [_Image.open(file_path).convert("RGB")]

        donut_result = None
        if use_donut and pages and extract_with_donut is not None and not use_pdfplumber_fallback:
            try:
                donut_result = extract_with_donut(pages[0])
            except Exception as exc:
                st.warning(f"Donut failed: {exc}")

        ocr_results = []
        for idx, page in enumerate(pages, start=1):
            if use_pdfplumber_fallback and isinstance(page, dict):
                ocr_results.append({
                    "page": page.get("page_number", idx),
                    "blocks": [{"text": page.get("text", ""), "confidence": None,
                                 "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]]}],
                })
            else:
                ocr_results.append({"page": idx, "blocks": ocr_image(page)})

    layout = parse_layout(
        ocr_results,
        donut_parsed=donut_result.get("parsed") if donut_result else None,
    )
    if pdfplumber_tables:
        for t in pdfplumber_tables:
            layout["tables"].append({"source": "pdfplumber", "row_count": len(t), "rows": t})

    extracted_text = layout.get("text", "")

    # NLP
    with st.spinner("Analysing…"):
        entities = extract_entities(extracted_text)
        key_phrases = extract_key_phrases(extracted_text)

    # Summary
    summary_text = None
    if use_summary:
        with st.spinner("Summarising…"):
            try:
                summary_text = summarize_text(extracted_text)
            except Exception:
                summary_text = None

    # ── results tabs ──
    st.header("📊 Extraction Results")
    tabs = st.tabs(["📝 Text", "💡 Summary", "🏷️ Entities", "📋 Tables", "🔧 JSON"])

    with tabs[0]:
        st.subheader("Extracted Text")
        if extracted_text.strip():
            st.text_area("Text", extracted_text, height=380)
            st.download_button(
                "⬇️ Download text",
                extracted_text,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.txt",
                mime="text/plain",
            )
        else:
            st.warning(
                "No text extracted. For scanned PDFs or images, install an OCR backend: "
                "`pip install easyocr` or `pip install pytesseract`"
            )

    with tabs[1]:
        st.subheader("Summary")
        if summary_text:
            st.write(summary_text)
            st.download_button(
                "⬇️ Download summary",
                summary_text,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt",
                mime="text/plain",
            )
        elif not use_summary:
            st.info("Enable 'Generate summary' in the sidebar.")
        else:
            st.info("No summary generated (document may be too short).")

    with tabs[2]:
        st.subheader("Detected Entities")
        if entities:
            st.caption(f"{len(entities)} entities found")
            st.dataframe(pd.DataFrame(entities), use_container_width=True)
        else:
            st.info("No entities detected.")
        if key_phrases:
            st.subheader("Key Phrases")
            st.write(", ".join(key_phrases[:40]))

    with tabs[3]:
        st.subheader("Detected Tables")
        tables = layout.get("tables", [])
        if not tables:
            st.info("No tables detected.")
        for i, table in enumerate(tables, start=1):
            rows = table.get("rows", [])
            st.write(f"**Table {i}** — {table.get('row_count', len(rows))} rows"
                     + (f" (source: {table.get('source')})" if table.get("source") else ""))
            if rows:
                try:
                    st.dataframe(pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows),
                                 use_container_width=True)
                except Exception:
                    st.table(rows)
                df_dl = pd.DataFrame(rows)
                st.download_button(
                    f"⬇️ Table {i} CSV",
                    df_dl.to_csv(index=False).encode("utf-8"),
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_table_{i}.csv",
                    mime="text/csv",
                )

    with tabs[4]:
        st.subheader("JSON Output")
        json_output = {
            "source_filename": uploaded_file.name,
            "text": extracted_text,
            "entities": entities,
            "key_phrases": key_phrases,
            "tables": layout.get("tables"),
            "summary": summary_text,
            "donut": donut_result,
        }
        st.json(json_output)
        st.download_button(
            "⬇️ Download JSON",
            json.dumps(json_output, indent=2),
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}.json",
            mime="application/json",
        )

    # ── save / search ──
    st.markdown("---")
    st.subheader("💾 Save to Database")
    title = st.text_input("Title (optional)", value=uploaded_file.name)
    if st.button("Save to Local DB"):
        pbar2 = st.progress(0)
        stxt2 = st.empty()

        def _cb2(cur, total, msg):
            pbar2.progress(cur / max(1, total))
            stxt2.text(msg)

        with st.spinner("Saving…"):
            result = _process_and_store(
                file_path, uploaded_file.name, storage, search,
                use_donut, use_trocr, use_summary, skip_duplicates,
                use_semantic_search=use_semantic_search and search is not None,
                progress_callback=_cb2,
                title=title,
            )
        status = result.get("status")
        if status == "skipped":
            st.info("Already ingested (skipped).")
        elif status == "saved":
            st.success("Saved to local database.")
        else:
            st.warning(f"Result: {status}")

    st.markdown("---")
    st.subheader("🔍 Search stored documents")
    col_q, col_k = st.columns([4, 1])
    with col_q:
        query_text_input = st.text_input("Search query", value="")
    with col_k:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)

    if not use_semantic_search:
        st.info("Enable 'Semantic search' in the sidebar to search stored documents.")
    elif search is None:
        st.warning("Search engine unavailable — install sentence-transformers or scikit-learn.")
    elif st.button("Search"):
        if not query_text_input.strip():
            st.warning("Enter a search query first.")
        else:
            with st.spinner("Searching…"):
                docs_with_emb = storage.get_documents_with_embeddings()
            if not docs_with_emb:
                st.warning("No documents with embeddings found. Save a document first.")
            else:
                import numpy as np
                candidates = []
                vecs = []
                for d in docs_with_emb:
                    try:
                        vecs.append(search.deserialize_embedding(d["embedding"]))
                        candidates.append(d)
                    except Exception:
                        pass

                if candidates:
                    results = search.search(
                        query_text_input, candidates,
                        np.vstack(vecs), top_k=int(top_k),
                    )
                    if not results:
                        st.info("No matching documents.")
                    for rank, r in enumerate(results, start=1):
                        with st.expander(
                            f"#{rank}  {r.get('title') or r.get('filename')}  "
                            f"(score: {r.get('score', 0):.3f})"
                        ):
                            st.caption(f"ID: {r.get('id')} | Saved: {r.get('created_at')}")
                            snippet = r.get("text", "")[:600]
                            st.markdown(_highlight_query(snippet, query_text_input))
                            if r.get("summary"):
                                st.markdown(f"**Summary:** {r.get('summary')}")
                else:
                    st.warning("No searchable documents found.")

    # ── compare / merge ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚖️ Compare documents")
    docs = storage.list_documents()
    doc_options = [f"{d['id']}: {d.get('title') or d.get('filename')}" for d in docs]

    if len(doc_options) >= 2:
        doc_a = st.sidebar.selectbox("Document A", doc_options, key="compare_a")
        doc_b = st.sidebar.selectbox("Document B", doc_options, key="compare_b")
        if st.sidebar.button("Compare"):
            id_a = int(doc_a.split(":", 1)[0])
            id_b = int(doc_b.split(":", 1)[0])
            doc_a_data = next((d for d in docs if d["id"] == id_a), None)
            doc_b_data = next((d for d in docs if d["id"] == id_b), None)
            if doc_a_data and doc_b_data:
                st.markdown("### Side-by-side")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**A: {doc_a_data.get('title') or doc_a_data.get('filename')}**")
                    st.text_area("Doc A", doc_a_data.get("text", ""), height=300, key="ta_a")
                with col2:
                    st.markdown(f"**B: {doc_b_data.get('title') or doc_b_data.get('filename')}**")
                    st.text_area("Doc B", doc_b_data.get("text", ""), height=300, key="ta_b")

                st.markdown("### Diff")
                st.markdown(_diff_text_md(doc_a_data.get("text", ""), doc_b_data.get("text", "")))

                st.markdown("### Repair / Merge")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("Repair A"):
                        src = (doc_a_data.get("metadata") or {}).get("source_path")
                        if src and os.path.exists(src):
                            res = _process_and_store(
                                src, doc_a_data.get("filename"), storage, search,
                                use_donut, use_trocr, use_summary, skip_duplicates=False,
                                use_semantic_search=use_semantic_search and search is not None,
                                title=doc_a_data.get("title"),
                            )
                            st.success(f"Repaired A: {res.get('status')}")
                        else:
                            st.warning("Source file not found for A.")
                with c2:
                    if st.button("Repair B"):
                        src = (doc_b_data.get("metadata") or {}).get("source_path")
                        if src and os.path.exists(src):
                            res = _process_and_store(
                                src, doc_b_data.get("filename"), storage, search,
                                use_donut, use_trocr, use_summary, skip_duplicates=False,
                                use_semantic_search=use_semantic_search and search is not None,
                                title=doc_b_data.get("title"),
                            )
                            st.success(f"Repaired B: {res.get('status')}")
                        else:
                            st.warning("Source file not found for B.")
                with c3:
                    if st.button("Merge A + B"):
                        merged_text = (doc_a_data.get("text", "") + "\n\n" + doc_b_data.get("text", ""))
                        merged_summary = " ".join(filter(None, [
                            doc_a_data.get("summary", ""), doc_b_data.get("summary", "")
                        ]))
                        new_title = (
                            f"Merged: {doc_a_data.get('title') or doc_a_data.get('filename')}"
                            f" + {doc_b_data.get('title') or doc_b_data.get('filename')}"
                        )
                        emb_blob = None
                        if search is not None:
                            try:
                                emb = search.embed([merged_text])[0]
                                emb_blob = search.serialize_embedding(emb)
                            except Exception:
                                pass
                        storage.add_document(
                            filename=new_title, file_hash=None, title=new_title,
                            text=merged_text, summary=merged_summary,
                            metadata={"merged_from": [doc_a_data["id"], doc_b_data["id"]]},
                            embedding=emb_blob,
                        )
                        st.success("Merged document created.")
    else:
        st.sidebar.write("Store at least 2 documents to compare.")

    # ── database panel ──
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗄️ Database")
    st.sidebar.write(f"Stored documents: **{len(docs)}**")
    for d in docs[:10]:
        st.sidebar.write(f"- [{d['id']}] {d.get('title') or d.get('filename')}")

    if st.sidebar.button("Export Excel report"):
        df = pd.DataFrame(docs)
        keep = [c for c in ["id", "filename", "title", "file_hash", "summary", "created_at"]
                if c in df.columns]
        buf = io.BytesIO()
        df[keep].to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        st.sidebar.download_button(
            "⬇️ Download report",
            buf,
            file_name="document_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if doc_options:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗑️ Delete document")
        del_choice = st.sidebar.selectbox("Select", doc_options, key="delete_doc")
        if st.sidebar.button("Delete selected"):
            del_id = int(del_choice.split(":", 1)[0])
            if storage.delete_document(del_id):
                st.sidebar.success(f"Deleted document {del_id}.")
            else:
                st.sidebar.error(f"Could not delete document {del_id}.")


if __name__ == "__main__":
    main()
