#!/usr/bin/env bash
# Builds everything the project needs, starting from a fresh clone:
#   1. gets ACTS (which contains traccc) at the commit the project was
#      developed with, if the acts/ folder is missing;
#   2. applies the local changes of patches/acts_local_changes.patch
#      (reco_cpu/reco_gpu targets, simulation);
#   3. configures traccc (only the first time, or with RECONFIGURE=1);
#   4. compiles the programs used by the scripts in exec/.
#
# Usage (from any folder):  ./exec/build.sh
#
# Settings, through environment variables (defaults in brackets):
#   CUDA_ROOT    CUDA installation  [/usr/local/cuda-13.0]
#                With no nvcc there the build is CPU only (no reco_gpu).
#   BOOST_ROOT   Boost installation [$HOME/local]
#   BUILD_DIR    build folder       [traccc-build]
#   JOBS         parallel jobs      [number of cores]
#   RECONFIGURE  1 = run the CMake configuration again [0]
# Example: CUDA_ROOT=/usr/local/cuda-12.4 JOBS=4 ./exec/build.sh
set -euo pipefail

# ACTS version the patch was made for.
ACTS_URL=https://github.com/acts-project/acts.git
ACTS_COMMIT=b4053980d9107a0b507e72f0f3e5720bc6d4b823

# Always work from the project folder, wherever the script is called from.
cd "$(dirname "$0")/.."
project=$(pwd)
patch_file=$project/patches/acts_local_changes.patch

CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-13.0}
BOOST_ROOT=${BOOST_ROOT:-$HOME/local}
BUILD_DIR=${BUILD_DIR:-traccc-build}
JOBS=${JOBS:-$(nproc)}
RECONFIGURE=${RECONFIGURE:-0}

step() { echo; echo "==> $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

# --- 1. ACTS ---------------------------------------------------------------
if [ ! -d acts/.git ]; then
    [ -e acts ] && fail "acts/ exists but is not a git clone: remove it and run again"
    step "Downloading ACTS at commit ${ACTS_COMMIT:0:12}"
    # Only the needed commit, without the whole history.
    git init -q acts
    git -C acts remote add origin "$ACTS_URL"
    git -C acts fetch --depth 1 origin "$ACTS_COMMIT"
    git -C acts checkout -q FETCH_HEAD
else
    current=$(git -C acts rev-parse HEAD)
    if [ "$current" != "$ACTS_COMMIT" ]; then
        echo "WARNING: acts/ is at commit ${current:0:12}, the patch was made for" \
             "${ACTS_COMMIT:0:12}: it may not apply" >&2
    fi
fi

# --- 2. Local changes to ACTS ----------------------------------------------
[ -f "$patch_file" ] || fail "missing $patch_file"
if git -C acts apply --reverse --check "$patch_file" 2>/dev/null; then
    step "Patch already applied"
elif git -C acts apply --check "$patch_file" 2>/dev/null; then
    step "Applying $(basename "$patch_file")"
    git -C acts apply "$patch_file"
else
    fail "the patch does not apply: acts/ has other local changes.
       See them with:      git -C acts diff --stat
       Discard them with:  git -C acts checkout -- .   (then run again)"
fi
if ! git -C acts diff | cmp -s - "$patch_file"; then
    echo "WARNING: acts/ contains changes that are not in the patch" \
         "(git -C acts diff --stat). To keep them, update the patch:" >&2
    echo "         git -C acts diff > patches/acts_local_changes.patch" >&2
fi

# --- 3. Configuration ------------------------------------------------------
nvcc=$CUDA_ROOT/bin/nvcc
if [ -x "$nvcc" ]; then
    with_cuda=1
    preset=cuda-fp32
    cuda_options=(
        "-DCMAKE_CUDA_COMPILER=$nvcc"
        "-DCUDAToolkit_ROOT=$CUDA_ROOT"
        "-DCMAKE_CUDA_ARCHITECTURES=native"
    )
    prefix_path="$BOOST_ROOT;$CUDA_ROOT"
else
    with_cuda=0
    preset=host-fp32
    cuda_options=()
    prefix_path="$BOOST_ROOT"
    echo "WARNING: no nvcc in $CUDA_ROOT: CPU-only build (set CUDA_ROOT to use the GPU)" >&2
fi

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ] || [ "$RECONFIGURE" = 1 ]; then
    step "Configuring traccc in $BUILD_DIR (preset $preset)"
    cmake --preset "$preset" -S acts/Traccc -B "$BUILD_DIR" \
        "${cuda_options[@]+"${cuda_options[@]}"}" \
        -DTRACCC_BUILD_ALPAKA=TRUE -DTRACCC_SETUP_ALPAKA=TRUE \
        -DTRACCC_USE_SYSTEM_ALPAKA=FALSE \
        "-DBoost_ROOT=$BOOST_ROOT" "-DCMAKE_PREFIX_PATH=$prefix_path" \
        -DACTS_USE_SYSTEM_EIGEN3=OFF \
        "-DACTS_CODEGEN_PREBUILT_DIR=$project/prebuilt-codegen"
else
    step "Configuration found in $BUILD_DIR (RECONFIGURE=1 to redo it)"
fi

# --- 4. Compilation --------------------------------------------------------
targets=(traccc_toy_mc traccc_reco_cpu traccc_throughput_mt_alpaka)
if [ "$with_cuda" = 1 ]; then
    targets+=(traccc_reco_gpu traccc_throughput_mt_cuda)
fi
step "Compiling ${targets[*]} with $JOBS jobs"
cmake --build "$BUILD_DIR" -j "$JOBS" --target "${targets[@]}"

step "Done. Programs in $BUILD_DIR/bin:"
for t in "${targets[@]}"; do
    ls -l "$BUILD_DIR/bin/$t" 2>/dev/null || echo "  (missing: $t)"
done