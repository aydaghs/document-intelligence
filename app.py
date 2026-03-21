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
        extract_entities,
        extract_key_phrases,
        ocr_image,
        ocr_pdf,
        parse_layout,
        summarize_text,
        file_hash,
        ensure_dir,
    )

    # Optional dependencies (may not install on all environments)
    try:
        from docintelligence import SemanticSearch
    except ImportError:
        SemanticSearch = None

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
    DocumentStorage = None  # type: ignore
    extract_entities = None  # type: ignore
    extract_key_phrases = None  # type: ignore
    ocr_image = None  # type: ignore
    ocr_pdf = None  # type: ignore
    parse_layout = None  # type: ignore
    summarize_text = None  # type: ignore
    file_hash = None  # type: ignore
    ensure_dir = None  # type: ignore
    SemanticSearch = None  # type: ignore
    extract_with_donut = None  # type: ignore
    trocr_ocr = None  # type: ignore


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "documents.db")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")


def _save_uploaded_file(uploaded_file) -> str:
    ensure_dir(UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


@st.cache_resource
def get_storage() -> DocumentStorage:
    ensure_dir(DATA_DIR)
    return DocumentStorage(DB_PATH)


@st.cache_resource
def get_search():
    if SemanticSearch is None:
        return None

    try:
        return SemanticSearch()
    except ImportError:
        # Sentence-transformers is not installed in this environment.
        return None
    except Exception as e:
        # Guard against any unexpected failure during semantic search initialization.
        st.warning(f"Semantic search unavailable: {e}")
        return None


def _build_query_text(extracted: dict) -> str:
    # Combine top-level text + table text into a single search input
    query_parts = [extracted.get("text", "")]
    for t in extracted.get("tables", []):
        for row in t.get("rows", []):
            query_parts.append(" ".join([str(c) for c in row]))
    return "\n".join(query_parts)


def _highlight_query(text: str, query: str) -> str:
    if not query or not text:
        return text

    def repl(match):
        return f"**{match.group(0)}**"

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(repl, text)


def _render_import_error() -> None:
    """Show import errors directly in Streamlit UI so Cloud logs are not required."""

    st.error("📦 Import error: required component failed to load.")
    st.markdown("**Error details:**")
    if IMPORT_ERROR is not None:
        st.code(traceback.format_exception_only(type(IMPORT_ERROR), IMPORT_ERROR)[-1].strip())
        st.write("---")
        st.text("Full traceback (for debugging):")
        st.code("\n".join(traceback.format_exception(type(IMPORT_ERROR), IMPORT_ERROR, IMPORT_ERROR.__traceback__)))
    else:
        st.write("No import error captured.")


def _require_imports() -> bool:
    if IMPORT_ERROR is not None:
        _render_import_error()
        return False
    return True


def _diff_text_html(a: str, b: str) -> str:
    """Generate an HTML diff between two texts."""

    differ = difflib.HtmlDiff(tabsize=4, wrapcolumn=80)
    return differ.make_table(a.splitlines(), b.splitlines(), context=True, numlines=3)


def _diff_text_md(a: str, b: str) -> str:
    """Generate a markdown diff (diff syntax highlighting)."""

    a_lines = a.splitlines()
    b_lines = b.splitlines()
    diff = list(difflib.unified_diff(a_lines, b_lines, lineterm=""))
    if not diff:
        return "No differences detected."
    return "\n".join(["```diff"] + diff + ["```"])


def _process_and_store(
    file_path: str,
    filename: str,
    storage: DocumentStorage,
    search: SemanticSearch,
    use_donut: bool,
    use_trocr: bool,
    use_summary: bool,
    skip_duplicates: bool,
    use_semantic_search: bool,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    title: Optional[str] = None,
) -> dict:
    """Process a file (OCR + layout + embeddings) and save it to storage."""

    from PIL import Image

    file_hash_val = file_hash(file_path)
    if skip_duplicates and storage.has_document_hash(file_hash_val):
        return {
            "status": "skipped",
            "reason": "already ingested",
            "file_hash": file_hash_val,
        }

    use_pdfplumber_fallback = False
    pdfplumber_tables = []
    pages = []
    if file_path.lower().endswith(".pdf"):
        try:
            from pdf2image import convert_from_path

            pages = convert_from_path(file_path, dpi=300)
        except Exception:
            # Poppler not available; fall back to pdfplumber text extraction
            import pdfplumber

            use_pdfplumber_fallback = True
            with pdfplumber.open(file_path) as pdf:
                for p in pdf.pages:
                    pages.append({"text": p.extract_text() or "", "page_number": p.page_number})
                    try:
                        # Attempt table extraction (may be empty)
                        tables = p.extract_tables()
                        if tables:
                            pdfplumber_tables.extend(tables)
                    except Exception:
                        pass
    else:
        pages = [Image.open(file_path).convert("RGB")]

    donut_result = None
    if use_donut and pages:
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
            ocr_results.append(
                {
                    "page": page.get("page_number", idx),
                    "blocks": [
                        {
                            "text": page.get("text", ""),
                            "confidence": None,
                            "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
                        }
                    ],
                }
            )
            continue

        if use_trocr:
            trocr_lines = trocr_ocr(page)
            blocks = [
                {"text": line, "confidence": None, "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]]}
                for line in trocr_lines
            ]
        else:
            blocks = ocr_image(page)
        ocr_results.append({"page": idx, "blocks": blocks})

    layout = parse_layout(
        ocr_results, donut_parsed=donut_result.get("parsed") if donut_result else None
    )

    # If we collected pdfplumber tables, add them to the layout output.
    if pdfplumber_tables:
        for t in pdfplumber_tables:
            layout["tables"].append({"source": "pdfplumber", "row_count": len(t), "rows": t})

    query_text = _build_query_text(layout)

    emb = None
    if use_semantic_search and search is not None:
        try:
            emb = search.embed([query_text])[0]
        except Exception:
            emb = None

    summary = None
    if use_summary:
        if summarize_text is None:
            summary = None
        else:
            try:
                summary = summarize_text(layout.get("text", ""))
            except Exception:
                summary = None

    storage.add_document(
        filename=filename,
        file_hash=file_hash_val,
        title=title or filename,
        text=query_text,
        summary=summary,
        metadata={"source_path": file_path},
        embedding=search.serialize_embedding(emb) if (search is not None and emb is not None) else None,
    )

    return {
        "status": "saved",
        "file_hash": file_hash_val,
        "layout": layout,
        "donut": donut_result,
        "summary": summary,
    }


def main() -> None:
    st.set_page_config(page_title="Document Intelligence", layout="wide")

    if not _require_imports():
        return

    st.title("Document Intelligence (OCR + Layout + Search)")

    st.sidebar.header("Upload & Extract")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
    )

    uploaded_files = st.sidebar.file_uploader(
        "Upload multiple files",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        accept_multiple_files=True,
    )

    use_donut = st.sidebar.checkbox("Use Donut (layout-aware parsing)", value=False)
    use_trocr = st.sidebar.checkbox("Use TrOCR (handwriting OCR)", value=False)
    use_summary = st.sidebar.checkbox("Generate summary (3 sentences)", value=True)
    skip_duplicates = st.sidebar.checkbox("Skip files already ingested (by hash)", value=True)
    preview_tables = st.sidebar.checkbox("Preview tables during ingestion", value=False)
    use_semantic_search = st.sidebar.checkbox(
        "Enable semantic search (sentence-transformers)", value=True
    )

    storage = get_storage()
    search = get_search()

    semantic_search_available = search is not None
    if use_semantic_search and not semantic_search_available:
        st.sidebar.warning(
            "Semantic search is not available because sentence-transformers is not installed. "
            "Install sentence-transformers and rebuild the app to enable it."
        )

    st.sidebar.markdown("### Batch ingest folder")
    ingest_folder = st.sidebar.text_input("Folder path (local)", value="")
    if st.sidebar.button("Ingest folder"):
        if not ingest_folder or not os.path.isdir(ingest_folder):
            st.sidebar.error("Please provide a valid folder path.")
        else:
            log_rows = []
            with st.sidebar.spinner("Ingesting folder..."):
                exts = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp")
                file_paths = []
                for root, _, files in os.walk(ingest_folder):
                    for fn in files:
                        if fn.lower().endswith(exts):
                            file_paths.append(os.path.join(root, fn))

                progress = st.sidebar.progress(0)
                for idx, fp in enumerate(file_paths):
                    try:
                        result = _process_and_store(
                            fp,
                            os.path.basename(fp),
                            storage,
                            search,
                            use_donut,
                            use_trocr,
                            use_summary,
                            skip_duplicates,
                            use_semantic_search=use_semantic_search and semantic_search_available,
                        )
                        status = result.get("status", "saved")
                        log_rows.append({
                            "file": fp,
                            "status": status,
                            "summary": result.get("summary"),
                            "reason": result.get("reason"),
                        })
                        if preview_tables and status == "saved":
                            # Show tables for first few files in sidebar
                            st.sidebar.write(f"### Preview: {os.path.basename(fp)}")
                            layout = result.get("layout") or {}
                            for t in layout.get("tables", []):
                                st.sidebar.write(t.get("rows", []))
                    except Exception as e:
                        log_rows.append({"file": fp, "status": "error", "reason": str(e)})
                    progress.progress((idx + 1) / max(1, len(file_paths)))

                st.sidebar.success(f"Completed ingest of {len(file_paths)} files.")

                # Show log table
                st.sidebar.markdown("---")
                st.sidebar.subheader("Ingest log")
                st.sidebar.table(log_rows)

    # Handle multiple file uploads (drag-and-drop folder style)
    if uploaded_files:
        st.write("### Processing uploaded files")
        file_log_rows = []
        for uf in uploaded_files:
            fp = _save_uploaded_file(uf)
            st.write(f"- Saved: {fp}")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _on_progress(cur: int, total: int, msg: str) -> None:
                progress_bar.progress(cur / max(1, total))
                status_text.text(msg)

            try:
                result = _process_and_store(
                    fp,
                    uf.name,
                    storage,
                    search,
                    use_donut,
                    use_trocr,
                    use_summary,
                    skip_duplicates,
                    use_semantic_search=use_semantic_search and semantic_search_available,
                    progress_callback=_on_progress,
                )
                file_log_rows.append({
                    "file": fp,
                    "status": result.get("status"),
                    "summary": result.get("summary"),
                    "reason": result.get("reason"),
                })
            except Exception as e:
                file_log_rows.append({"file": fp, "status": "error", "reason": str(e)})

        st.markdown("---")
        st.subheader("Upload log")
        st.table(file_log_rows)

    if uploaded_file is None:
        st.info("Upload a PDF or scanned image to start extraction.")
    else:
        file_path = _save_uploaded_file(uploaded_file)
        st.success(f"Saved uploaded file: {file_path}")

        if file_path.lower().endswith(".pdf"):
            st.info(
                "PDF extraction on Streamlit Cloud uses a text-only fallback (no Poppler). "
                "For full OCR/table extraction, run this app locally with Poppler installed (e.g. `apt-get install poppler-utils`) or deploy to an environment with Poppler."
            )

            # Try to render a preview of the first page (pdfplumber requires no Poppler for basic rendering)
            try:
                import pdfplumber

                with pdfplumber.open(file_path) as pdf:
                    if pdf.pages:
                        pil_img = pdf.pages[0].to_image(resolution=150).original
                        st.image(pil_img, caption="PDF preview (page 1)", use_column_width=True)
            except Exception:
                pass

        pages = []
        donut_result = None

        with st.spinner("Running OCR..."):
            if uploaded_file.type == "application/pdf" or file_path.lower().endswith(".pdf"):
                try:
                    from pdf2image import convert_from_path

                    pages = convert_from_path(file_path, dpi=300)
                    use_pdfplumber_fallback = False
                except Exception as e:
                    # Poppler is not available on the Streamlit Cloud environment.
                    # Fall back to pdfplumber-based text extraction.
                    import pdfplumber

                    use_pdfplumber_fallback = True
                    pages = []
                    with pdfplumber.open(file_path) as pdf:
                        for p in pdf.pages:
                            pages.append({"text": p.extract_text() or "", "page_number": p.page_number})
            else:
                from PIL import Image

                pages = [Image.open(file_path).convert("RGB")]
                use_pdfplumber_fallback = False

            if use_donut and pages and not use_pdfplumber_fallback:
                try:
                    donut_result = extract_with_donut(pages[0])
                except Exception as exc:
                    st.warning(
                        "Donut extraction failed (check model download and transformers install): %s" % exc
                    )

            # Fallback to EasyOCR for structure extraction
            ocr_results = []
            for idx, page in enumerate(pages, start=1):
                if use_pdfplumber_fallback:
                    # pdfplumber provides text but no OCR boxes
                    ocr_results.append(
                        {
                            "page": page.get("page_number", idx),
                            "blocks": [{
                                "text": page.get("text", ""),
                                "confidence": None,
                                # Provide a dummy bbox so that layout parsing includes this block.
                                "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
                            }],
                        }
                    )
                else:
                    ocr_results.append({"page": idx, "blocks": ocr_image(page)})

        layout = parse_layout(
            ocr_results, donut_parsed=donut_result.get("parsed") if donut_result else None
        )

        if extract_entities is None:
            entities = []
            st.warning(
                "Entity extraction is disabled because spaCy is not installed. "
                "Install spaCy to enable named entity recognition."
            )
        else:
            entities = extract_entities(layout.get("text", ""))

        # Summary generation (for display) - this runs independently of saving
        summary_text = None
        if use_summary:
            if summarize_text is None:
                summary_text = None
            else:
                with st.spinner("Generating summary..."):
                    try:
                        summary_text = summarize_text(layout.get("text", ""))
                    except Exception:
                        summary_text = None

        st.header("Extraction Results")

        tabs = st.tabs(["Text", "Summary", "Entities", "Tables", "JSON"])

        with tabs[0]:
            st.subheader("Extracted Text")
            text_content = layout.get("text", "")
            st.text_area("Text", text_content, height=360)
            st.download_button(
                "Download text",
                text_content,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.txt",
                mime="text/plain",
            )

        with tabs[1]:
            st.subheader("Summary")
            if not use_summary:
                st.info("Enable 'Generate summary' in the sidebar to see this.")
            elif summarize_text is None:
                st.warning(
                    "Summarization is unavailable (missing transformers). "
                    "Install transformers and restart to enable."
                )
            elif summary_text:
                st.write(summary_text)
                st.download_button(
                    "Download summary",
                    summary_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt",
                    mime="text/plain",
                )
            else:
                st.info("No summary generated.")

        with tabs[2]:
            st.subheader("Detected Entities")
            if not entities:
                st.info("No entities detected. Install spaCy and run `python -m spacy download en_core_web_sm` to enable NER.")
            else:
                st.caption(f"{len(entities)} entities found")
                st.table(entities[:30])
            key_phrases_list = []
            if extract_key_phrases is not None:
                try:
                    key_phrases_list = extract_key_phrases(layout.get("text", ""))
                except Exception:
                    pass
            if key_phrases_list:
                st.subheader("Key Phrases")
                st.write(", ".join(key_phrases_list[:40]))

        with tabs[3]:
            st.subheader("Detected Tables")
            tables = layout.get("tables", [])
            if not tables:
                st.info("No tables detected.")
            for i, table in enumerate(tables, start=1):
                st.write(f"**Table {i} (rows={table.get('row_count')})**")
                st.table(table.get("rows", []))
                df = pd.DataFrame(table.get("rows", []))
                st.download_button(
                    f"Download Table {i} as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_table_{i}.csv",
                    mime="text/csv",
                )

        with tabs[4]:
            st.subheader("JSON Output")
            json_output = {
                "source_filename": uploaded_file.name,
                "text": layout.get("text"),
                "entities": entities,
                "tables": layout.get("tables"),
                "donut": donut_result,
            }
            st.json(json_output)
            st.download_button(
                "Download JSON",
                json.dumps(json_output, indent=2),
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}.json",
                mime="application/json",
            )

        st.markdown("---")
        st.subheader("Save / Search")
        title = st.text_input("Title (optional)", value=uploaded_file.name)
        if st.button("Save to Local DB"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _on_progress(cur: int, total: int, msg: str) -> None:
                progress_bar.progress(cur / max(1, total))
                status_text.text(msg)

            with st.spinner("Saving document..."):
                result = _process_and_store(
                    file_path,
                    uploaded_file.name,
                    storage,
                    search,
                    use_donut,
                    use_trocr,
                    use_summary,
                    skip_duplicates,
                    use_semantic_search=use_semantic_search and semantic_search_available,
                    progress_callback=_on_progress,
                )
            status = result.get("status")
            if status == "skipped":
                st.info("Document was already ingested (skipped).")
            elif status == "saved":
                st.success("Saved document to local datastore.")
            else:
                st.warning(f"Document processing result: {status}")

        st.markdown("---")
        st.subheader("Search stored documents")
        col_q, col_k = st.columns([4, 1])
        with col_q:
            query_text = st.text_input("Search query", value="")
        with col_k:
            top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
        if not use_semantic_search:
            st.info("Semantic search is disabled. Enable it in the sidebar to search stored documents.")
        elif search is None:
            st.warning(
                "Semantic search is unavailable (missing sentence-transformers). "
                "Install sentence-transformers and restart to enable."
            )
        elif st.button("Search"):
            if not query_text.strip():
                st.warning("Enter a search query first.")
            else:
                with st.spinner("Searching..."):
                    docs_with_emb = storage.get_documents_with_embeddings()
                if not docs_with_emb:
                    st.warning("No previously stored documents found. Save a document first.")
                else:
                    import numpy as np

                    candidate_embeddings = []
                    candidates = []
                    for d in docs_with_emb:
                        emb_blob = d.get("embedding")
                        if emb_blob is None:
                            continue
                        try:
                            candidate_embeddings.append(search.deserialize_embedding(emb_blob))
                            candidates.append(d)
                        except Exception:
                            pass
                    if candidate_embeddings:
                        candidate_embeddings = np.vstack(candidate_embeddings)
                        results = search.search(query_text, candidates, candidate_embeddings, top_k=int(top_k))
                        if not results:
                            st.info("No matching documents found.")
                        for rank, r in enumerate(results, start=1):
                            with st.expander(
                                f"#{rank} — {r.get('title') or r.get('filename')}  (score: {r.get('score', 0):.3f})"
                            ):
                                st.caption(f"ID: {r.get('id')} | Created: {r.get('created_at')}")
                                snippet = r.get("text", "")[:600]
                                st.markdown(_highlight_query(snippet, query_text))
                                if r.get("summary"):
                                    st.markdown(f"**Summary:** {r.get('summary')}")
                    else:
                        st.warning("No document embeddings found. Please save a document with embedding.")

    # Document comparison / change detection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Compare stored documents")
    docs = storage.list_documents()
    doc_options = [f"{d['id']}: {d.get('title') or d.get('filename')}" for d in docs]

    if len(doc_options) >= 2:
        doc_a = st.sidebar.selectbox("Document A", doc_options, key="compare_a")
        doc_b = st.sidebar.selectbox("Document B", doc_options, key="compare_b")
        if st.sidebar.button("Compare documents"):
            id_a = int(doc_a.split(":", 1)[0])
            id_b = int(doc_b.split(":", 1)[0])
            doc_a_data = next((d for d in docs if d["id"] == id_a), None)
            doc_b_data = next((d for d in docs if d["id"] == id_b), None)
            if doc_a_data and doc_b_data:
                st.markdown("### Side-by-side view")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**A: {doc_a_data.get('title') or doc_a_data.get('filename')}**")
                    st.text_area("Doc A text", doc_a_data.get("text", ""), height=300)
                with col2:
                    st.markdown(f"**B: {doc_b_data.get('title') or doc_b_data.get('filename')}**")
                    st.text_area("Doc B text", doc_b_data.get("text", ""), height=300)

                st.markdown("### Diff (markdown)")
                diff_md = _diff_text_md(doc_a_data.get("text", ""), doc_b_data.get("text", ""))
                st.markdown(diff_md)

                # Repair/merge UI
                st.markdown("### Repair / Merge")
                if st.button("Repair Document A"):
                    # Reprocess and overwrite document A using its source path
                    meta = doc_a_data.get("metadata", {})
                    source = meta.get("source_path")
                    if source and os.path.exists(source):
                        res = _process_and_store(
                            source,
                            doc_a_data.get("filename"),
                            storage,
                            search,
                            use_donut,
                            use_trocr,
                            use_summary,
                            skip_duplicates=False,
                            use_semantic_search=use_semantic_search and semantic_search_available,
                            title=doc_a_data.get("title"),
                        )
                        st.success(f"Repaired document A: {res.get('status')}")
                    else:
                        st.warning("Original source file not found for document A.")
                if st.button("Repair Document B"):
                    meta = doc_b_data.get("metadata", {})
                    source = meta.get("source_path")
                    if source and os.path.exists(source):
                        res = _process_and_store(
                            source,
                            doc_b_data.get("filename"),
                            storage,
                            search,
                            use_donut,
                            use_trocr,
                            use_summary,
                            skip_duplicates=False,
                            use_semantic_search=use_semantic_search and semantic_search_available,
                            title=doc_b_data.get("title"),
                        )
                        st.success(f"Repaired document B: {res.get('status')}")
                    else:
                        st.warning("Original source file not found for document B.")

                if st.button("Merge A + B into new document"):
                    merged_text = doc_a_data.get("text", "") + "\n\n" + doc_b_data.get("text", "")
                    merged_summary = "".join(
                        filter(None, [doc_a_data.get("summary", ""), doc_b_data.get("summary", "")])
                    )
                    new_title = f"Merged: {doc_a_data.get('title') or doc_a_data.get('filename')} + {doc_b_data.get('title') or doc_b_data.get('filename')}"
                    emb = None
                    emb_blob = None
                    if search is not None:
                        try:
                            emb = search.embed([merged_text])[0]
                            emb_blob = search.serialize_embedding(emb)
                        except Exception:
                            emb_blob = None
                    storage.add_document(
                        filename=new_title,
                        file_hash=None,
                        title=new_title,
                        text=merged_text,
                        summary=merged_summary,
                        metadata={"merged_from": [doc_a_data["id"], doc_b_data["id"]]},
                        embedding=emb_blob,
                    )
                    st.success("Merged document created.")
            else:
                st.warning("Unable to load selected documents for comparison.")
    else:
        st.sidebar.write("Store at least 2 documents to compare.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Database")
    st.sidebar.write(f"Stored documents: {len(docs)}")
    for d in docs[:10]:
        st.sidebar.write(f"- [{d['id']}] {d.get('title') or d.get('filename')} ({d.get('created_at')})")

    if st.sidebar.button("Export Excel report"):
        df = pd.DataFrame(docs)
        keep_cols = [c for c in ["id", "filename", "title", "file_hash", "summary", "created_at"] if c in df.columns]
        df = df[keep_cols]
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        st.sidebar.download_button(
            label="Download report",
            data=buf,
            file_name="document_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if doc_options:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Delete a document")
        del_choice = st.sidebar.selectbox("Select document to delete", doc_options, key="delete_doc")
        if st.sidebar.button("Delete selected document"):
            del_id = int(del_choice.split(":", 1)[0])
            deleted = storage.delete_document(del_id)
            if deleted:
                st.sidebar.success(f"Deleted document {del_id}.")
            else:
                st.sidebar.error(f"Could not delete document {del_id}.")


if __name__ == "__main__":
    main()
