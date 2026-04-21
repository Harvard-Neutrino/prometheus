import json
import time
import logging

from prometheus.summary import emit_run_summary
from prometheus import config


class DummyLogCounter:
    def __init__(self):
        self.warning_count = 0
        self.error_count = 0
        self.critical_count = 0
        self.info_count = 0


class FakeEvent:
    def __init__(self):
        self.final_states = ["MuMinus"]


class FakeProm:
    def __init__(self):
        self.injection = [FakeEvent()]
        class Detector:
            n_modules = 42
            modules = list(range(42))
        self.detector = Detector()
        # timing attributes
        self._start_inj = 0.0
        self._end_inj = 0.1
        self._start_prop = 0.2
        self._end_prop = 1.0
        self._start_out = 1.0
        self._end_out = 1.5
        self._run_start_time = 0.0
        # captured noise
        self._captured_warnings = []
        self._init_output = ""
        self._inject_output = ""
        self._propagate_output = ""
        self._log_counter = DummyLogCounter()


def test_emit_run_summary_user_and_json(tmp_path, monkeypatch, capsys):
    outfile = tmp_path / "out.parquet"
    outfile.write_bytes(b"dummy")

    # Configure run preferences
    monkeypatch.setattr(config.run, "summary_mode", "user")
    monkeypatch.setattr(config.run, "compact", False)
    monkeypatch.setattr(config.run, "summary_json", True)
    json_path = tmp_path / "summary.json"
    monkeypatch.setattr(config.run, "summary_json_path", str(json_path))
    monkeypatch.setattr(config.run, "run_number", 7)
    monkeypatch.setattr(config.run, "nevents", 1)

    prom = FakeProm()
    emit_run_summary(prom, str(outfile), time.time(), size=None)

    captured = capsys.readouterr()
    assert "📦 Output ready" in captured.out or "Output ready" in captured.out

    # JSON summary must have been written
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["run_number"] == 7
    assert data["output"]["path"] == str(outfile)


def test_emit_run_summary_compact_mode(tmp_path, monkeypatch, capsys):
    outfile = tmp_path / "out.parquet"
    outfile.write_bytes(b"x")

    monkeypatch.setattr(config.run, "summary_mode", "user")
    monkeypatch.setattr(config.run, "compact", True)
    monkeypatch.setattr(config.run, "summary_json", False)

    prom = FakeProm()
    emit_run_summary(prom, str(outfile), time.time(), size=None)
    captured = capsys.readouterr()
    # compact single-line should contain pipes
    assert "|" in captured.out


def test_emit_run_summary_debug_mode_logs(tmp_path, monkeypatch, caplog):
    outfile = tmp_path / "out.parquet"
    outfile.write_bytes(b"y")

    monkeypatch.setattr(config.run, "summary_mode", "debug")
    monkeypatch.setattr(config.run, "compact", False)
    monkeypatch.setattr(config.run, "summary_json", False)

    prom = FakeProm()
    caplog.set_level(logging.DEBUG)
    emit_run_summary(prom, str(outfile), time.time(), size=None)

    # Look for the debug summary entry
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Run debug summary" in m or "Checksum (sha256)" in m or "Timings [s]" in m for m in msgs)
