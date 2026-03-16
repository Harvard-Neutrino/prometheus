import shutil
import os

def clean_ppc_tmpdir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
