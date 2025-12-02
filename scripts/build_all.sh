#!/usr/bin/env bash
# Build all plugins in the repository
# Usage: ./scripts/build_all.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGINS_DIR="$REPO_ROOT/plugins"

echo "Building all vLLM plugins..."
echo "Repository root: $REPO_ROOT"
echo ""

# Build shared utilities first
if [ -d "$REPO_ROOT/shared" ]; then
    echo "=== Building shared utilities ==="
    cd "$REPO_ROOT/shared"
    python -m build
    echo ""
fi

# Build each plugin
for plugin_dir in "$PLUGINS_DIR"/*/; do
    if [ -f "$plugin_dir/pyproject.toml" ]; then
        plugin_name=$(basename "$plugin_dir")
        echo "=== Building $plugin_name ==="
        cd "$plugin_dir"
        python -m build
        echo ""
    fi
done

echo "All plugins built successfully!"
echo "Built packages are in each plugin's dist/ directory"
