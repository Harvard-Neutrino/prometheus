#!/usr/bin/env bash
# scripts/install_hooks.sh — install local git hooks for this repository.
#
# Run once after cloning:
#   bash scripts/install_hooks.sh
#
# Currently installs:
#   pre-push  — strips Jupyter notebook outputs via nbstripout before pushing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="$REPO_ROOT/.git/hooks"

if [[ ! -d "$HOOKS_DIR" ]]; then
    echo "ERROR: $HOOKS_DIR not found. Is this a git repository?" >&2
    exit 1
fi

if ! command -v nbstripout &>/dev/null; then
    echo "nbstripout not found — installing..."
    pip install --quiet nbstripout
fi

# ── pre-push hook ─────────────────────────────────────────────────────────────
cat > "$HOOKS_DIR/pre-push" << 'EOF'
#!/usr/bin/env bash
# Strip Jupyter notebook outputs before pushing.
# Aborts the push and commits the stripped files if any outputs were present.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

mapfile -t notebooks < <(find . -name '*.ipynb' -not -path '*/.git/*')

if [[ ${#notebooks[@]} -eq 0 ]]; then
    exit 0
fi

dirty=()
for nb in "${notebooks[@]}"; do
    if ! nbstripout --check "$nb" 2>/dev/null; then
        dirty+=("$nb")
    fi
done

if [[ ${#dirty[@]} -eq 0 ]]; then
    exit 0
fi

echo ""
echo "The following notebooks have stored outputs:"
for nb in "${dirty[@]}"; do
    echo "  $nb"
done
echo ""
echo "Stripping outputs now..."
nbstripout "${dirty[@]}"
git add "${dirty[@]}"
git commit -m "chore: strip notebook outputs before push"
echo "Done. Outputs stripped and committed. Re-run 'git push' to continue."
echo ""
# Exit 1 so git re-evaluates the push with the new commit included.
exit 1
EOF

chmod +x "$HOOKS_DIR/pre-push"

echo "Installed pre-push hook -> $HOOKS_DIR/pre-push"
echo "Notebook outputs will be stripped automatically before each push."
