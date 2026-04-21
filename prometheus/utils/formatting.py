"""Formatting and filesystem helpers.

Small utilities for size formatting and file checksums.
"""
import hashlib


def _file_checksum(path: str, algo: str = "sha256") -> str:
    """Return hex digest of *path* using *algo*; empty string on error."""
    try:
        h = hashlib.new(algo)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _human_size(num: int) -> str:
    """Return a human-readable size for *num* bytes."""
    try:
        n = float(num)
    except Exception:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


__all__ = ["_file_checksum", "_human_size"]
