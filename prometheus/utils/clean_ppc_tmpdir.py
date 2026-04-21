from pathlib import Path

def clean_ppc_tmpdir(path):
    p = Path(path)
    if p.is_dir():
        for f in p.glob("*"):
            try:
                if f.is_file():
                    f.unlink()
            except Exception:
                pass
        try:
            p.rmdir()
        except Exception:
            pass
    else:
        try:
            p.unlink()
        except Exception:
            pass
