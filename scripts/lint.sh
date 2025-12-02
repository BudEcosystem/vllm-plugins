#!/usr/bin/env bash
# Run linting on all plugins
# Usage: ./scripts/lint.sh [--fix]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIX_FLAG=""

if [ "$1" = "--fix" ]; then
    FIX_FLAG="--fix"
fi

echo "Running linter on all vLLM plugins..."
echo ""

# Run ruff on the entire repo
cd "$REPO_ROOT"

echo "=== Running ruff ==="
if [ -n "$FIX_FLAG" ]; then
    ruff check $FIX_FLAG plugins/ shared/
    ruff format plugins/ shared/
else
    ruff check plugins/ shared/
    ruff format --check plugins/ shared/
fi

echo ""
echo "=== Running mypy ==="
mypy plugins/ shared/ --ignore-missing-imports || true

echo ""
echo "Linting complete!"
