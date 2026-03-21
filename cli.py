import argparse
import logging
import os
from typing import Optional

from docintelligence import (
    DocumentStorage,
    ocr_image,
    ocr_pdf,
    parse_layout,
    file_hash,
)

# Optional dependencies — imported safely so CLI still works without them
try:
    from docintelligence import SemanticSearch
except ImportError:
    SemanticSearch = None  # type: ignore

try:
    from docintelligence import summarize_text
except ImportError:
    summarize_text = None  # type: ignore

try:
    from docintelligence import extract_with_donut
except ImportError:
    extract_with_donut = None  # type: ignore

try:
    from docintelligence import trocr_ocr
except ImportError:
    trocr_ocr = None  # type: ignore


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "documents.db")

_LOG = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _process_file(
    file_path: str,
    title: str,
    storage: DocumentStorage,
    search: Optional[object],
    use_donut: bool,
    use_trocr: bool,
    use_summary: bool,
    skip_duplicates: bool,
    overwrite_id: Optional[int] = None,
) -> dict:
    """Process a single file and write to storage (or update existing document)."""

    file_hash_val = file_hash(file_path)
    if skip_duplicates and storage.get_document_by_hash(file_hash_val):
        return {"status": "skipped", "reason": "already ingested"}

    # Run Donut (layout-aware OCR) on first page if requested
    donut_result = None
    if use_donut and extract_with_donut is not None:
        try:
            from PIL import Image
            img = Image.open(file_path).convert("RGB")
            donut_result = extract_with_donut(img)
            _LOG.info("Donut extraction succeeded for %s", file_path)
        except Exception as exc:
            _LOG.warning("Donut extraction failed for %s: %s", file_path, exc)

    # Convert file to OCR page data
    pages = []
    if file_path.lower().endswith(".pdf"):
        try:
            pages = ocr_pdf(file_path)
        except Exception as exc:
            _LOG.warning("ocr_pdf failed for %s: %s — skipping", file_path, exc)
            return {"status": "error", "reason": str(exc)}
    else:
        try:
            from PIL import Image
            pil_img = Image.open(file_path).convert("RGB")
            if use_trocr and trocr_ocr is not None:
                lines = trocr_ocr(pil_img)
                blocks = [
                    {"text": line, "confidence": None, "bbox": [[0, 0], [0, 0], [0, 0], [0, 0]]}
                    for line in lines
                ]
            else:
                blocks = ocr_image(pil_img)
            pages = [{"page": 1, "blocks": blocks}]
        except Exception as exc:
            _LOG.warning("OCR failed for %s: %s — skipping", file_path, exc)
            return {"status": "error", "reason": str(exc)}

    # Layout parsing — pass Donut output when available
    layout = parse_layout(
        pages,
        donut_parsed=donut_result.get("parsed") if donut_result else None,
    )
    text = layout.get("text", "")

    summary = None
    if use_summary:
        if summarize_text is not None:
            try:
                summary = summarize_text(text)
            except Exception as exc:
                _LOG.warning("Summarization failed for %s: %s", file_path, exc)
        else:
            _LOG.warning("summarize_text not available — skipping summarization")

    emb_blob = None
    if search is not None:
        try:
            emb = search.embed([text])[0]
            emb_blob = search.serialize_embedding(emb)
        except Exception as exc:
            _LOG.warning("Embedding failed for %s: %s", file_path, exc)

    if overwrite_id is not None:
        storage.update_document(
            overwrite_id,
            title,
            text,
            summary,
            {"source_path": file_path},
            emb_blob,
        )
        return {"status": "updated", "id": overwrite_id}

    storage.add_document(
        filename=os.path.basename(file_path),
        file_hash=file_hash_val,
        title=title,
        text=text,
        summary=summary,
        metadata={"source_path": file_path},
        embedding=emb_blob,
    )
    return {"status": "saved"}


def ingest_folder(
    folder: str,
    use_donut: bool,
    use_trocr: bool,
    use_summary: bool,
    skip_duplicates: bool,
) -> None:
    _ensure_dir(DATA_DIR)
    storage = DocumentStorage(DB_PATH)

    search = None
    if SemanticSearch is not None:
        try:
            search = SemanticSearch()
        except Exception as exc:
            _LOG.warning("Semantic search unavailable: %s — embeddings will be skipped", exc)

    exts = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    file_paths = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                file_paths.append(os.path.join(root, fn))

    print(f"Found {len(file_paths)} file(s) to process.")
    for fp in file_paths:
        try:
            result = _process_file(
                fp,
                title=os.path.basename(fp),
                storage=storage,
                search=search,
                use_donut=use_donut,
                use_trocr=use_trocr,
                use_summary=use_summary,
                skip_duplicates=skip_duplicates,
            )
            print(f"  {fp} → {result.get('status')}"
                  + (f" ({result.get('reason')})" if result.get("reason") else ""))
        except Exception as exc:
            _LOG.error("Unexpected error processing %s: %s", fp, exc)
            print(f"  {fp} → error: {exc}")


def list_documents() -> None:
    _ensure_dir(DATA_DIR)
    storage = DocumentStorage(DB_PATH)
    docs = storage.list_documents()
    if not docs:
        print("No documents stored.")
        return
    for d in docs:
        print(f"{d['id']:>4}  {d.get('title') or d.get('filename')}  (hash={d.get('file_hash')})")


def repair_document(doc_id: int) -> None:
    _ensure_dir(DATA_DIR)
    storage = DocumentStorage(DB_PATH)

    search = None
    if SemanticSearch is not None:
        try:
            search = SemanticSearch()
        except Exception as exc:
            _LOG.warning("Semantic search unavailable: %s", exc)

    doc = storage.get_document_by_id(doc_id)
    if not doc:
        print(f"Document {doc_id} not found.")
        return

    meta = doc.get("metadata", {})
    src = meta.get("source_path")
    if not src or not os.path.exists(src):
        print(f"Source file not available for document {doc_id}: {src!r}")
        return

    result = _process_file(
        src,
        title=doc.get("title") or doc.get("filename"),
        storage=storage,
        search=search,
        use_donut=False,
        use_trocr=False,
        use_summary=True,
        skip_duplicates=False,
        overwrite_id=doc_id,
    )
    print(f"Repaired document {doc_id}: {result.get('status')}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Document Intelligence CLI")
    parser.add_argument("--ingest-folder", help="Ingest a folder of documents")
    parser.add_argument("--list", action="store_true", help="List stored documents")
    parser.add_argument("--repair", type=int, help="Reprocess an existing document by ID")
    parser.add_argument("--use-donut", action="store_true", help="Use Donut for layout parsing")
    parser.add_argument("--use-trocr", action="store_true", help="Use TrOCR for handwriting")
    parser.add_argument("--no-summary", action="store_true", help="Disable summarization")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip files already ingested by hash")

    args = parser.parse_args()

    if args.list:
        list_documents()
        return

    if args.repair is not None:
        repair_document(args.repair)
        return

    if args.ingest_folder:
        ingest_folder(
            args.ingest_folder,
            use_donut=args.use_donut,
            use_trocr=args.use_trocr,
            use_summary=not args.no_summary,
            skip_duplicates=args.skip_duplicates,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
