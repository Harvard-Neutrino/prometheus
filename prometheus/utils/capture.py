"""Native output capture utilities.

This module provides a context manager to capture low-level (C/C++)
stdout/stderr writes that bypass Python's ``sys.stdout`` redirection.
"""

from typing import Tuple


class _COutputCapture:
    """Context manager to capture C-level stdout/stderr (file-descriptor).

    This captures writes emitted via native extensions (e.g., std::cout)
    which are not intercepted by Python's ``sys.stdout`` redirectors.
    """

    def __enter__(self):
        import os

        # Use raw file descriptor numbers for robustness under test runners
        # which may replace sys.stdout/sys.stderr objects. FD 1 and 2 are the
        # standard stdout and stderr descriptors at the OS level.
        self._stdout_fd = 1
        self._stderr_fd = 2
        self._saved_stdout = os.dup(self._stdout_fd)
        self._saved_stderr = os.dup(self._stderr_fd)
        self._r_out, self._w_out = os.pipe()
        self._r_err, self._w_err = os.pipe()
        os.dup2(self._w_out, self._stdout_fd)
        os.dup2(self._w_err, self._stderr_fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        import os
        import sys

        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(self._saved_stdout, self._stdout_fd)
        os.dup2(self._saved_stderr, self._stderr_fd)
        try:
            os.close(self._saved_stdout)
        except Exception:
            pass
        try:
            os.close(self._saved_stderr)
        except Exception:
            pass
        # close write ends so readers see EOF
        try:
            os.close(self._w_out)
        except Exception:
            pass
        try:
            os.close(self._w_err)
        except Exception:
            pass
        try:
            with os.fdopen(self._r_out, "rb") as ro:
                self.out = ro.read().decode("utf-8", errors="replace")
        except Exception:
            self.out = ""
        try:
            with os.fdopen(self._r_err, "rb") as re:
                self.err = re.read().decode("utf-8", errors="replace")
        except Exception:
            self.err = ""
        return False


def capture_native_output(func, *args, **kwargs) -> Tuple[str, str]:
    """Run ``func`` while capturing native stdout/stderr and return (out, err).

    Useful for tests or short-lived calls where capturing native prints is
    desirable.
    """
    with _COutputCapture() as cap:
        func(*args, **kwargs)
    return cap.out, cap.err


__all__ = ["_COutputCapture", "capture_native_output"]
