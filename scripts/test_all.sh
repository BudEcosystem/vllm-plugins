#!/usr/bin/env bash
# Run tests for all plugins
# Usage: ./scripts/test_all.sh [pytest-args]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGINS_DIR="$REPO_ROOT/plugins"
PYTEST_ARGS="${@:-}"

echo "Running tests for all vLLM plugins..."
echo "Repository root: $REPO_ROOT"
echo ""

# Track results
PASSED=()
FAILED=()

# Test each plugin
for plugin_dir in "$PLUGINS_DIR"/*/; do
    if [ -d "$plugin_dir/tests" ]; then
        plugin_name=$(basename "$plugin_dir")
        echo "=== Testing $plugin_name ==="
        cd "$plugin_dir"

        # Install in development mode if not already
        pip install -e ".[dev]" --quiet 2>/dev/null || true

        if pytest tests/ $PYTEST_ARGS; then
            PASSED+=("$plugin_name")
        else
            FAILED+=("$plugin_name")
        fi
        echo ""
    fi
done

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Passed: ${#PASSED[@]}"
for p in "${PASSED[@]}"; do
    echo "  ✓ $p"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed: ${#FAILED[@]}"
    for f in "${FAILED[@]}"; do
        echo "  ✗ $f"
    done
    exit 1
fi

echo ""
echo "All tests passed!"
