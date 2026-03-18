import hashlib
import os


def ensure_dir(path: str) -> None:
    """Ensure a directory exists (creates it if needed)."""

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def file_hash(path: str, algorithm: str = "sha1") -> str:
    """Compute a file hash (used to dedupe ingested documents)."""

    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
