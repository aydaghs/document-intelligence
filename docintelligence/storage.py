from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentStorage:
    """SQLite-backed storage for extracted documents and their text chunks."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        c = self.conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_hash TEXT,
                title TEXT,
                text TEXT,
                summary TEXT,
                metadata TEXT,
                embedding BLOB,
                created_at TEXT NOT NULL
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB
            )"""
        )
        # Backwards-compatible column additions
        for col, col_def in [
            ("file_hash", "TEXT"),
            ("summary", "TEXT"),
            ("category", "TEXT"),
            ("category_confidence", "REAL"),
            ("category_meta", "TEXT"),
        ]:
            try:
                c.execute(f"ALTER TABLE documents ADD COLUMN {col} {col_def}")
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

    # ── documents ─────────────────────────────────────────────────────────────

    def add_document(
        self,
        filename: str,
        file_hash: Optional[str],
        title: Optional[str],
        text: str,
        summary: Optional[str],
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[bytes],
        category: Optional[str] = None,
        category_confidence: Optional[float] = None,
        category_meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        c = self.conn.cursor()
        c.execute(
            """INSERT INTO documents
               (filename, file_hash, title, text, summary, metadata, embedding,
                category, category_confidence, category_meta, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filename,
                file_hash,
                title,
                text,
                summary,
                json.dumps(metadata or {}),
                embedding,
                category,
                category_confidence,
                json.dumps(category_meta or {}),
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        return c.lastrowid  # type: ignore[return-value]

    def _row_to_doc(self, row) -> Dict[str, Any]:
        return {
            "id": row[0],
            "filename": row[1],
            "file_hash": row[2],
            "title": row[3],
            "text": row[4],
            "summary": row[5],
            "metadata": json.loads(row[6] or "{}"),
            "embedding": row[7],
            "category": row[8],
            "category_confidence": row[9],
            "category_meta": json.loads(row[10] or "{}"),
            "created_at": row[11],
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            """SELECT id, filename, file_hash, title, text, summary, metadata,
                      embedding, category, category_confidence, category_meta, created_at
               FROM documents ORDER BY created_at DESC"""
        )
        return [self._row_to_doc(r) for r in c.fetchall()]

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            """SELECT id, filename, file_hash, title, text, summary, metadata,
                      embedding, category, category_confidence, category_meta, created_at
               FROM documents WHERE file_hash = ?""",
            (file_hash,),
        )
        row = c.fetchone()
        return self._row_to_doc(row) if row else None

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            """SELECT id, filename, file_hash, title, text, summary, metadata,
                      embedding, category, category_confidence, category_meta, created_at
               FROM documents WHERE id = ?""",
            (doc_id,),
        )
        row = c.fetchone()
        return self._row_to_doc(row) if row else None

    def has_document_hash(self, file_hash: str) -> bool:
        return self.get_document_by_hash(file_hash) is not None

    def update_document(
        self,
        doc_id: int,
        title: Optional[str],
        text: str,
        summary: Optional[str],
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[bytes],
        category: Optional[str] = None,
        category_confidence: Optional[float] = None,
        category_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        c = self.conn.cursor()
        c.execute(
            """UPDATE documents
               SET title = ?, text = ?, summary = ?, metadata = ?, embedding = ?,
                   category = ?, category_confidence = ?, category_meta = ?
               WHERE id = ?""",
            (
                title,
                text,
                summary,
                json.dumps(metadata or {}),
                embedding,
                category,
                category_confidence,
                json.dumps(category_meta or {}),
                doc_id,
            ),
        )
        self.conn.commit()

    def get_documents_with_embeddings(self) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            """SELECT id, filename, file_hash, title, text, summary, metadata,
                      embedding, category, category_confidence, category_meta, created_at
               FROM documents WHERE embedding IS NOT NULL ORDER BY created_at DESC"""
        )
        return [self._row_to_doc(r) for r in c.fetchall()]

    def delete_document(self, doc_id: int) -> bool:
        c = self.conn.cursor()
        # Cascade via FK; also delete chunks explicitly for safety
        c.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return c.rowcount > 0

    # ── chunks ─────────────────────────────────────────────────────────────────

    def save_chunks(
        self,
        doc_id: int,
        chunks: List[str],
        embeddings: Optional[List[bytes]] = None,
    ) -> None:
        """Store text chunks (and optional per-chunk embeddings) for a document."""
        c = self.conn.cursor()
        c.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        for i, text in enumerate(chunks):
            emb = embeddings[i] if embeddings and i < len(embeddings) else None
            c.execute(
                "INSERT INTO chunks (doc_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
                (doc_id, i, text, emb),
            )
        self.conn.commit()

    def get_chunks(self, doc_id: int) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            "SELECT id, doc_id, chunk_index, text, embedding FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        )
        return [
            {"id": r[0], "doc_id": r[1], "chunk_index": r[2], "text": r[3], "embedding": r[4]}
            for r in c.fetchall()
        ]

    def has_chunks(self, doc_id: int) -> bool:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,))
        return c.fetchone()[0] > 0

    # ── legacy helpers ─────────────────────────────────────────────────────────

    def get_embeddings(self) -> List[bytes]:
        c = self.conn.cursor()
        c.execute("SELECT embedding FROM documents WHERE embedding IS NOT NULL")
        return [row[0] for row in c.fetchall() if row[0] is not None]

    def close(self) -> None:
        self.conn.close()
