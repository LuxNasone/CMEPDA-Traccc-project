#!/usr/bin/env bash
# Builds everything the project needs, starting from a fresh clone:
#   1. gets ACTS (which contains traccc) at the commit the project was
#      developed with, if the acts/ folder is missing; ACTS is used as it
#      is, without changes;
#   2. configures traccc (only the first time, or with RECONFIGURE=1);
#   3. compiles traccc and installs it in traccc-install/ (the programs of
#      the project use it as an external library);
#   4. compiles the programs of the project (src/CMakeLists.txt: simulation
#      and reconstruction) in build/, with the executables in build/bin.
#
# The first run compiles the whole of traccc (also its tests and examples,
# needed by the installation): it takes a while. The next runs only
# recompile what changed.
#
# Usage (from any folder):  ./exec/build.sh
#
# Settings, through environment variables (defaults in brackets):
#   CUDA_ROOT    CUDA installation  [/usr/local/cuda-13.0]
#                With no nvcc there the build is CPU only (no reco_gpu).
#   BOOST_ROOT   Boost installation [$HOME/local]
#   BUILD_DIR    build folder of traccc        [traccc-build]
#   INSTALL_DIR  installation folder of traccc [traccc-install]
#   PROJECT_BUILD_DIR  build folder of the project programs  [build]
#   JOBS         parallel jobs      [number of cores]
#   RECONFIGURE  1 = run the CMake configuration again [0]
# Example: CUDA_ROOT=/usr/local/cuda-12.4 JOBS=4 ./exec/build.sh
set -euo pipefail

# ACTS version the project was developed with.
ACTS_URL=https://github.com/acts-project/acts.git
ACTS_COMMIT=b4053980d9107a0b507e72f0f3e5720bc6d4b823

# Always work from the project folder, wherever the script is called from.
cd "$(dirname "$0")/.."
project=$(pwd)

CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-13.0}
BOOST_ROOT=${BOOST_ROOT:-$HOME/local}
BUILD_DIR=${BUILD_DIR:-traccc-build}
INSTALL_DIR=${INSTALL_DIR:-$project/traccc-install}
PROJECT_BUILD_DIR=${PROJECT_BUILD_DIR:-build}
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
        echo "WARNING: acts/ is at commit ${current:0:12}, the project was" \
             "developed with ${ACTS_COMMIT:0:12}" >&2
    fi
fi
# The project uses ACTS without changes: warn if there are any
if [ -n "$(git -C acts status --porcelain --untracked-files=no)" ]; then
    echo "WARNING: acts/ has local changes, the project expects it unmodified" \
         "(see them with: git -C acts diff --stat)" >&2
fi

# --- 2. Configuration ------------------------------------------------------
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
        "-DACTS_CODEGEN_PREBUILT_DIR=$project/prebuilt-codegen" \
        "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR"
else
    step "Configuration found in $BUILD_DIR (RECONFIGURE=1 to redo it)"
    # The installation folder must be fixed in the configuration: ACTS writes
    # some files to CMAKE_INSTALL_PREFIX and ignores "cmake --install --prefix"
    if ! grep -qx "CMAKE_INSTALL_PREFIX:PATH=$INSTALL_DIR" "$BUILD_DIR/CMakeCache.txt"; then
        step "Setting the installation folder of traccc to $INSTALL_DIR"
        cmake -S acts/Traccc -B "$BUILD_DIR" "-DCMAKE_INSTALL_PREFIX=$INSTALL_DIR"
    fi
fi

# --- 3. traccc: compilation and installation -------------------------------
# Everything, not only some targets: the installation copies all of traccc
step "Compiling traccc in $BUILD_DIR with $JOBS jobs"
cmake --build "$BUILD_DIR" -j "$JOBS"
step "Installing traccc in $INSTALL_DIR"
cmake --install "$BUILD_DIR" > /dev/null

# --- 4. Programs of the project ---------------------------------------------
if [ "$with_cuda" = 1 ]; then build_gpu=ON; else build_gpu=OFF; fi
step "Configuring and compiling src/ in $PROJECT_BUILD_DIR"
cmake -S src -B "$PROJECT_BUILD_DIR" \
    "-DCMAKE_PREFIX_PATH=$INSTALL_DIR;$prefix_path" \
    "-DRECO_BUILD_GPU=$build_gpu"
cmake --build "$PROJECT_BUILD_DIR" -j "$JOBS"

step "Done. Programs in $PROJECT_BUILD_DIR/bin:"
programs=(traccc_toy_mc traccc_reco_cpu traccc_hough_reco)
if [ "$with_cuda" = 1 ]; then
    programs+=(traccc_reco_gpu)
fi
for program in "${programs[@]}"; do
    ls -l "$PROJECT_BUILD_DIR/bin/$program" 2>/dev/null || echo "  (missing: $program)"
done