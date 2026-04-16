"""Compatibility helpers for optional/removed backends.

This package contains small utilities used during the Phase 7 migration
to keep test fixtures and unpickling working without requiring the full
`dm-haiku` dependency at runtime.
"""

__all__ = ["haiku_unpickler"]
