"""Pytest configuration for the prometheus test suite.

Changes the working directory to the repository root before any test runs so
that all relative paths (geo files, resource files) resolve consistently whether
pytest is invoked from the repo root or from inside tests/.
"""
import os
import pathlib

# Repo root is one level above this file.
REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()

def pytest_configure(config):
    os.chdir(REPO_ROOT)
