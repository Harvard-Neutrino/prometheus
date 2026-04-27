import os

from prometheus.utils.capture import _COutputCapture, capture_native_output


def test_coutputcapture_fd_writes():
    cap = _COutputCapture()
    cap.__enter__()
    try:
        # write directly to file descriptors 1 and 2
        os.write(1, b"HELLO_STDOUT\n")
        os.write(2, b"HELLO_STDERR\n")
    finally:
        cap.__exit__(None, None, None)

    assert "HELLO_STDOUT" in cap.out
    assert "HELLO_STDERR" in cap.err


def test_capture_native_output_helper():
    def _emit():
        os.write(1, b"X_OUT\n")
        os.write(2, b"X_ERR\n")

    out, err = capture_native_output(_emit)
    assert "X_OUT" in out
    assert "X_ERR" in err
