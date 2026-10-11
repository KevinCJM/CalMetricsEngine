#!/bin/sh
# Preparation only. Does not build, install, import or execute numerical work.
set -eu

CME_SCREEN_DRIVER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CME_SCREEN_ORIGINAL_ROOT=${CME_SCREEN_ORIGINAL_ROOT:-/Users/chenjunming/Desktop/KevinGit/CppCalAST/CalMetricsEngine}
CME_SCREEN_TARGET=${1:-/private/tmp/calmetrics-mc-projection-integration/eigen-screen/source}

case "$CME_SCREEN_TARGET" in
  /private/tmp/*) ;;
  *) echo "screen mirror must be a new directory under /private/tmp" >&2; exit 2 ;;
esac
if [ -e "$CME_SCREEN_TARGET" ]; then
  echo "refusing to overwrite existing mirror: $CME_SCREEN_TARGET" >&2
  exit 2
fi
test -f "$CME_SCREEN_ORIGINAL_ROOT/CMakeLists.txt"
test -f "$CME_SCREEN_DRIVER_ROOT/candidate-affine_projection.cpp"
CME_SCREEN_BASE_SHA=$(shasum -a 256 "$CME_SCREEN_ORIGINAL_ROOT/cpp/affine_projection.cpp" | awk '{print $1}')
if [ "$CME_SCREEN_BASE_SHA" != 77bb7f37ae21abdf065078a2397da0d3ed88029363bc1f3d37f489bb2699c665 ]; then
  echo "production baseline changed; review the candidate delta before a new mirror" >&2
  exit 3
fi
mkdir -p "$CME_SCREEN_TARGET"
rsync -a \
  --exclude='.git' --exclude='.codegraph' --exclude='.venv' \
  --exclude='.env*' --exclude='.agents' --exclude='.codex' --exclude='.ruff_cache' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.mypy_cache' \
  --exclude='/build/' --exclude='/dist/' --exclude='/install/' --exclude='/_skbuild/' \
  --exclude='/docs/physical-execution-evidence/' \
  "$CME_SCREEN_ORIGINAL_ROOT/" "$CME_SCREEN_TARGET/"
cp "$CME_SCREEN_DRIVER_ROOT/candidate-affine_projection.cpp" "$CME_SCREEN_TARGET/cpp/affine_projection.cpp"
printf '%s\n' "$CME_SCREEN_ORIGINAL_ROOT" > "$CME_SCREEN_TARGET/.screen-original-root"
printf '%s\n' "$CME_SCREEN_BASE_SHA" > "$CME_SCREEN_TARGET/.screen-original-affine-sha256"
shasum -a 256 "$CME_SCREEN_TARGET/cpp/affine_projection.cpp"
printf 'CME_SCREEN_SOURCE_ROOT=%s\n' "$CME_SCREEN_TARGET"
printf '%s\n' 'Mirror prepared only; root must build, install, create a fresh identity manifest and run serially.'
