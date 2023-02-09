import os
from glob import glob

def clean_ppc_tmpdir(path):
    if os.path.isdir(path):
        fs = glob(f"{path}/*")
        for f in fs:
            os.remove(f)
        os.rmdir(path)
    else:
        os.remove(path)
