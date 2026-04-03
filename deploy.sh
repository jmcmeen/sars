#!/usr/bin/env bash
# deploy.sh — build and publish sars to PyPI
# Usage:
#   ./deploy.sh          → publish to PyPI (production)
#   ./deploy.sh --test   → publish to TestPyPI first (dry run)
#
# Requires: pip install build twine
# Requires: .env file with PYPI_API_TOKEN set (see .env.example)

set -euo pipefail

# ── Load environment ──────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  echo "Error: .env file not found. Copy .env.example and fill in your token."
  exit 1
fi
export $(grep -v '^#' .env | xargs)

if [ -z "${PYPI_API_TOKEN:-}" ]; then
  echo "Error: PYPI_API_TOKEN not set in .env"
  exit 1
fi

# ── Parse args ────────────────────────────────────────────────────────────────
TEST_MODE=false
if [ "${1:-}" = "--test" ]; then
  TEST_MODE=true
  echo "→ Test mode: publishing to TestPyPI"
else
  echo "→ Production mode: publishing to PyPI"
fi

# ── Preflight checks ──────────────────────────────────────────────────────────
echo ""
echo "── Preflight ────────────────────────────────────────────────────────────"

# Confirm on the right branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ] && [ "$TEST_MODE" = false ]; then
  echo "Error: production deploys must be from main (currently on '$BRANCH')"
  exit 1
fi

# Confirm clean working tree
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Error: uncommitted changes present. Commit or stash before deploying."
  exit 1
fi

# Run tests
echo "Running test suite..."
pip install -e ".[dev]" -q
pytest --tb=short -q
echo "Tests passed."

# Lint
echo "Running ruff..."
ruff check src/
echo "Lint passed."

# ── Get version ───────────────────────────────────────────────────────────────
VERSION=$(python -c "import tomllib; f=open('pyproject.toml','rb'); d=tomllib.load(f); print(d['project']['version'])")
echo ""
echo "── Building sars v${VERSION} ─────────────────────────────────────────────"

# Confirm with user on production deploy
if [ "$TEST_MODE" = false ]; then
  read -p "Publish sars v${VERSION} to PyPI? [y/N] " confirm
  if [ "${confirm}" != "y" ] && [ "${confirm}" != "Y" ]; then
    echo "Aborted."
    exit 0
  fi
fi

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo "── Building ─────────────────────────────────────────────────────────────"
rm -rf dist/ build/
python -m build
echo "Build complete:"
ls -lh dist/

# ── Publish ───────────────────────────────────────────────────────────────────
echo ""
echo "── Publishing ───────────────────────────────────────────────────────────"

if [ "$TEST_MODE" = true ]; then
  TWINE_PASSWORD="${PYPI_API_TOKEN}" twine upload \
    --repository testpypi \
    --username __token__ \
    dist/*
  echo ""
  echo "✓ Published to TestPyPI."
  echo "  Install with: pip install --index-url https://test.pypi.org/simple/ sars==${VERSION}"
else
  TWINE_PASSWORD="${PYPI_API_TOKEN}" twine upload \
    --username __token__ \
    dist/*
  echo ""
  echo "✓ Published to PyPI."
  echo "  Install with: pip install sars==${VERSION}"

  # Tag the release
  if ! git tag | grep -q "v${VERSION}"; then
    git tag "v${VERSION}"
    git push origin "v${VERSION}"
    echo "  Git tag v${VERSION} pushed."
  fi
fi

echo ""
echo "Done."
