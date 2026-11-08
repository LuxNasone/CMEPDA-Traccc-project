### CMEPDA Traccc performance project

### Brief description

The project hosted and developed in this repository is a test of the performance of the GPU usage of the Traccc library. 

### Requirements

  - CMake 3.25 or above
  - C++ compiler 20 or above
  - ACTS library
  - CUDA 12.6 or above
  - Boost 1.78 or above

### Command line options

- CPU
  sh
  cmake --preset alpaka-fp32 -S acts/Traccc -B traccc-build
  cmake --build traccc-build --target traccc_toy_mc traccc_seq_example_alpaka \
  traccc_throughput_mt_alpaka -j 2

- GPU
  sh
  cmake --preset cuda-fp32 -S acts/Traccc -B traccc-build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DTRACCC_BUILD_ALPAKA=TRUE -DTRACCC_SETUP_ALPAKA=TRUE -DTRACCC_USE_SYSTEM_ALPAKA=FALSE \
  -DBoost_ROOT=$HOME/local -DCMAKE_PREFIX_PATH="$HOME/local;/usr/local/cuda-13.0" \
  -DACTS_USE_SYSTEM_EIGEN3=OFF \
  -DACTS_CODEGEN_PREBUILT_DIR=$PWD/prebuilt-codegen
  cmake --build traccc-build --target traccc_toy_mc traccc_seq_example_cuda \
  traccc_throughput_mt_cuda traccc_seq_example_alpaka traccc_throughput_mt_alpaka -j 2

- Server environmental variables (in `~/.bashrc`):
  sh
  export PATH=/usr/local/cuda-13.0/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$HOME/local/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$HOME/Exam_2/traccc-build/gnu_13.3_cxx20_64_relwithdebinfo:$LD_LIBRARY_PATH

- Simulation
  sh
  export TRACCC_TEST_DATA_DIR=$PWD/data
  mkdir -p data/NOME_RUN && cd data/NOME_RUN
  ../../traccc-build/bin/traccc_toy_mc ../../config/toy_mc.conf
  cd ../..

- Digitization
  sh
  python src/digitization/toy_digitizer.py data/NOME_RUN --config config/digitization.conf

- Reconstruction
  sh
  D=$PWD/data/NOME_RUN
  traccc-build/bin/traccc_seq_example_alpaka \
  --detector-file=$D/toy_detector_geometry.json \
  --material-file=$D/toy_detector_homogeneous_material.json \
  --grid-file=$D/toy_detector_surface_grids.json \
  --digitization-file=$D/toy_digitization.json \
  --conditions-file=$D/toy_digitization.json \
  --use-acts-geom-source=false \
  --input-directory=$D/ \
  --input-events=1 \
  --seedfinder-r-range=25:200 \
  --finding-run-smoother=none \
  --constraint-step-size-mm=1 --search-window 3:3

Also use --use-acts-geom-source=false, --seedfinder-r-range=25:200, --finding-run-smoother=none

- Throughput 
  ./run_throughput.sh cuda data/NOME_RUN
  ./run_throughput.sh alpaka data/NOME_RUN --cpu-threads=8