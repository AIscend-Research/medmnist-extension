#!/usr/bin/env bash
# One-command local gate: lint, unit tests, then the full synthetic-data
# end-to-end pipeline. Run this before every commit.
#
# This is the check that would have caught the topfer.py sklearn bug before
# it reached a submitted result: pytest alone did not, because
# tool.pytest.ini_options silences UserWarning and the bug only surfaced as
# an exception swallowed by a bare `except`, not a test failure.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ruff (correctness rules only) =="
ruff check cartomnist scripts tests

echo
echo "== pytest =="
python -m pytest tests/ -q

echo
echo "== smoke test (synthetic data, CPU, ~3 min) =="
python scripts/smoke_test.py

echo
echo "ALL CHECKS PASSED"
