#!/usr/bin/env bash
# Install all plugins in development mode
# Usage: ./scripts/install_dev.sh

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLUGINS_DIR="$REPO_ROOT/plugins"

echo "Installing all vLLM plugins in development mode..."
echo "Repository root: $REPO_ROOT"
echo ""

# Install shared utilities first
if [ -d "$REPO_ROOT/shared" ]; then
    echo "=== Installing shared utilities ==="
    pip install -e "$REPO_ROOT/shared"
    echo ""
fi

# Install each plugin
for plugin_dir in "$PLUGINS_DIR"/*/; do
    if [ -f "$plugin_dir/pyproject.toml" ]; then
        plugin_name=$(basename "$plugin_dir")
        echo "=== Installing $plugin_name ==="
        pip install -e "$plugin_dir[dev]"
        echo ""
    fi
done

echo "All plugins installed in development mode!"
echo ""
echo "Verify installation with:"
echo "  pip list | grep vllm"
