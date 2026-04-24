from pathlib import Path

def clean_ppc_tmpdir(path):
    """Remove the ppc temporary directory and all files within it.

    Parameters
    ----------
    path : str or Path
        Path to the ppc temporary directory to remove.
    """
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
