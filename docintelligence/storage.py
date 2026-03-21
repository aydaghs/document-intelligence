from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DocumentStorage:
    """Simple SQLite-backed storage for extracted documents."""

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
        # Ensure backwards compatibility: add missing columns if needed
        for column in ["file_hash", "summary"]:
            try:
                c.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

    def add_document(
        self,
        filename: str,
        file_hash: Optional[str],
        title: Optional[str],
        text: str,
        summary: Optional[str],
        metadata: Optional[Dict[str, Any]],
        embedding: Optional[bytes],
    ) -> int:
        """Add a document record and return its ID."""

        c = self.conn.cursor()
        c.execute(
            "INSERT INTO documents (filename, file_hash, title, text, summary, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                filename,
                file_hash,
                title,
                text,
                summary,
                json.dumps(metadata or {}),
                embedding,
                datetime.utcnow().isoformat(),
            ),
        )
        self.conn.commit()
        return c.lastrowid

    def list_documents(self) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            "SELECT id, filename, file_hash, title, text, summary, metadata, created_at FROM documents ORDER BY created_at DESC"
        )
        rows = c.fetchall()
        return [
            {
                "id": r[0],
                "filename": r[1],
                "file_hash": r[2],
                "title": r[3],
                "text": r[4],
                "summary": r[5],
                "metadata": json.loads(r[6] or "{}"),
                "created_at": r[7],
            }
            for r in rows
        ]

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            "SELECT id, filename, file_hash, title, text, summary, metadata, embedding, created_at FROM documents WHERE file_hash = ?",
            (file_hash,),
        )
        row = c.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "filename": row[1],
            "file_hash": row[2],
            "title": row[3],
            "text": row[4],
            "summary": row[5],
            "metadata": json.loads(row[6] or "{}"),
            "embedding": row[7],
            "created_at": row[8],
        }

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            "SELECT id, filename, file_hash, title, text, summary, metadata, embedding, created_at FROM documents WHERE id = ?",
            (doc_id,),
        )
        row = c.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "filename": row[1],
            "file_hash": row[2],
            "title": row[3],
            "text": row[4],
            "summary": row[5],
            "metadata": json.loads(row[6] or "{}"),
            "embedding": row[7],
            "created_at": row[8],
        }

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
    ) -> None:
        c = self.conn.cursor()
        c.execute(
            "UPDATE documents SET title = ?, text = ?, summary = ?, metadata = ?, embedding = ? WHERE id = ?",
            (
                title,
                text,
                summary,
                json.dumps(metadata or {}),
                embedding,
                doc_id,
            ),
        )
        self.conn.commit()

    def get_embeddings(self) -> List[bytes]:
        c = self.conn.cursor()
        c.execute("SELECT embedding FROM documents WHERE embedding IS NOT NULL")
        return [row[0] for row in c.fetchall() if row[0] is not None]

    def get_documents_with_embeddings(self) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute(
            "SELECT id, filename, file_hash, title, text, summary, metadata, embedding, created_at FROM documents WHERE embedding IS NOT NULL ORDER BY created_at DESC"
        )
        rows = c.fetchall()
        docs: List[Dict[str, Any]] = []
        for r in rows:
            docs.append(
                {
                    "id": r[0],
                    "filename": r[1],
                    "file_hash": r[2],
                    "title": r[3],
                    "text": r[4],
                    "summary": r[5],
                    "metadata": json.loads(r[6] or "{}"),
                    "embedding": r[7],
                    "created_at": r[8],
                }
            )
        return docs

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document by ID. Returns True if a row was deleted."""
        c = self.conn.cursor()
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self.conn.commit()
        return c.rowcount > 0

    def close(self) -> None:
        self.conn.close()
