"""Hashing utilities for change detection."""

import hashlib
from pathlib import Path
from typing import List, Optional


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex string of the SHA256 hash.
    """
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_directory_hash(path: Path, extensions: Optional[List[str]] = None) -> str:
    """Compute a hash representing a directory's content.

    Args:
        path: Directory path.
        extensions: File extensions to include (default: .py, .toml, .cfg).

    Returns:
        SHA256 hash of concatenated file hashes.
    """
    if extensions is None:
        extensions = [".py", ".toml", ".cfg", ".txt", ".md"]

    hasher = hashlib.sha256()

    # Sort for deterministic ordering
    try:
        files = sorted(path.rglob("*"))
    except Exception:
        return hashlib.sha256(str(path).encode()).hexdigest()

    for file_path in files:
        if file_path.is_file():
            # Skip hidden files and __pycache__
            if any(part.startswith(".") or part == "__pycache__" for part in file_path.parts):
                continue

            # Only hash relevant extensions
            if extensions and file_path.suffix not in extensions:
                continue

            # Include relative path in hash for structure changes
            try:
                rel_path = file_path.relative_to(path)
                hasher.update(str(rel_path).encode())
                hasher.update(compute_file_hash(file_path).encode())
            except (ValueError, OSError):
                pass

    return hasher.hexdigest()


def compute_string_hash(content: str) -> str:
    """Compute SHA256 hash of a string.

    Args:
        content: String to hash.

    Returns:
        Hex string of the SHA256 hash (first 16 chars).
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]
