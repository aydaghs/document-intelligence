import argparse
import os
from typing import Optional

from docintelligence import (
    DocumentStorage,
    SemanticSearch,
    extract_with_donut,
    ocr_image,
    ocr_pdf,
    parse_layout,
    summarize_text,
    trocr_ocr,
    file_hash,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "documents.db")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _process_file(
    file_path: str,
    title: str,
    storage: DocumentStorage,
    search: SemanticSearch,
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

    # convert PDF -> images if needed.
    pages = []
    if file_path.lower().endswith(".pdf"):
        pages = ocr_pdf(file_path)
    else:
        from PIL import Image

        pages = [{"page": 1, "blocks": ocr_image(Image.open(file_path).convert("RGB"))}]

    # layout parsing
    layout = parse_layout(pages)
    text = layout.get("text", "")

    summary = None
    if use_summary:
        summary = summarize_text(text)

    emb = search.embed([text])[0]

    if overwrite_id is not None:
        storage.update_document(
            overwrite_id,
            title,
            text,
            summary,
            {"source_path": file_path},
            search.serialize_embedding(emb),
        )
        return {"status": "updated", "id": overwrite_id}

    storage.add_document(
        filename=os.path.basename(file_path),
        file_hash=file_hash_val,
        title=title,
        text=text,
        summary=summary,
        metadata={"source_path": file_path},
        embedding=search.serialize_embedding(emb),
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
    search = SemanticSearch()

    exts = (".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp")
    file_paths = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                file_paths.append(os.path.join(root, fn))

    for fp in file_paths:
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
        print(fp, result)


def list_documents() -> None:
    _ensure_dir(DATA_DIR)
    storage = DocumentStorage(DB_PATH)
    docs = storage.list_documents()
    for d in docs:
        print(f"{d['id']} - {d['filename']} (hash={d.get('file_hash')})")


def repair_document(doc_id: int) -> None:
    _ensure_dir(DATA_DIR)
    storage = DocumentStorage(DB_PATH)
    search = SemanticSearch()

    doc = storage.get_document_by_id(doc_id)
    if not doc:
        print(f"Document {doc_id} not found.")
        return

    meta = doc.get("metadata", {})
    src = meta.get("source_path")
    if not src or not os.path.exists(src):
        print(f"Source file not available for document {doc_id}.")
        return

    _process_file(
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
    print(f"Repaired document {doc_id}.")


def main() -> None:
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
