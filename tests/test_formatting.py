import hashlib
import tempfile

from prometheus.utils.formatting import _file_checksum, _human_size


def test_human_size():
    assert _human_size(0).startswith("0.0")
    assert _human_size(1024).startswith("1.0 KB")


def test_file_checksum(tmp_path):
    p = tmp_path / "data.bin"
    data = b"hello-checksum"
    p.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    got = _file_checksum(str(p))
    assert got == expected
