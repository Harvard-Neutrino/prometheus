"""Pytest configuration for the prometheus test suite.

Changes the working directory to the repository root before any test runs so
that all relative paths (geo files, resource files) resolve consistently whether
pytest is invoked from the repo root or from inside tests/.
"""

import os
import pathlib

import pytest

# Repo root is one level above this file.
REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()


def pytest_configure(config):
    os.chdir(REPO_ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow end-to-end simulation tests (may take several minutes)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        skip = pytest.mark.skip(reason="slow e2e test; pass --run-slow to enable")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip)
