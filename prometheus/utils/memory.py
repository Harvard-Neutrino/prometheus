"""Helpers for returning propagator memory to the operating system."""

import ctypes
import gc
import logging

logger = logging.getLogger(__name__)

_MALLOC_TRIM = None
_MALLOC_TRIM_LOOKED_UP = False


def _malloc_trim() -> bool:
    """Ask glibc to return free heap arenas to the operating system.

    Returns
    -------
    bool
        True when ``malloc_trim`` was available and called.
    """
    global _MALLOC_TRIM, _MALLOC_TRIM_LOOKED_UP
    if not _MALLOC_TRIM_LOOKED_UP:
        _MALLOC_TRIM_LOOKED_UP = True
        try:
            libc = ctypes.CDLL("libc.so.6")
            _MALLOC_TRIM = libc.malloc_trim
        except (OSError, AttributeError):
            # Not glibc (musl, macOS). The JAX and Python frees below still
            # happen, they are just not handed back to the OS as promptly.
            _MALLOC_TRIM = None
    if _MALLOC_TRIM is None:
        return False
    _MALLOC_TRIM(0)
    return True


def release_propagator_memory() -> None:
    """Drop the compiled-executable cache and return the heap to the OS.

    The olympus photon propagator pads its kernel inputs to power-of-two
    buckets, so XLA compiles and caches one executable per bucket size
    reached. Those executables live for the lifetime of the process, and on a
    densely instrumented detector they accumulate to well over a gigabyte.
    Every worker holds its own copy, so this is what decides how many workers
    fit in RAM.

    Two steps are needed and neither is sufficient alone: ``jax.clear_caches``
    frees the executables, and ``malloc_trim`` hands the resulting free arenas
    back to the OS. Without the trim the resident set barely moves and the
    cache misleadingly looks as though it was holding nothing.

    The cost is that the next propagation re-compiles the shapes it uses,
    which is cheap when the persistent on-disk compilation cache is enabled
    (``jax_compilation_cache_dir``) since that turns recompilation into a
    cache read. Enable it before relying on this in a throughput-sensitive
    loop.

    Live ``jax.Array`` objects are untouched -- only compiled code is dropped,
    so simulation results are unchanged.
    """
    try:
        import jax
    except ImportError:
        # Nothing to release when the olympus path was never imported.
        return

    try:
        jax.clear_caches()
    except Exception:
        logger.debug("jax.clear_caches() failed; continuing", exc_info=True)

    gc.collect()
    _malloc_trim()
