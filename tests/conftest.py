"""Skip the tests of a task nobody has started yet, so pull requests stay green.

A task counts as "started" once its owner deletes every "TODO An" line in its file.
Until then its tests show as SKIPPED (not failed). The grading script counts
skipped tests as not done.
"""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "milkcheck"
FILES = {"A1": "data.py", "A2": "stats.py", "A3": "model.py", "A4": "api.py", "A5": "evaluate.py"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        m = re.search(r"test_(a[1-5])_", item.nodeid)
        if not m:
            continue
        task = m.group(1).upper()
        f = SRC / FILES[task]
        if f"TODO {task}" in f.read_text(encoding="utf-8"):
            item.add_marker(pytest.mark.skip(
                reason=f"{task} not started: delete the 'TODO {task}' lines in src/milkcheck/{FILES[task]} when you begin"))
