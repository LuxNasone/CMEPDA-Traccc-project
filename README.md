### CMEPDA Traccc performance project

### Brief description

The project hosted and developed in this repository is a test of the performance of the GPU usage of the Traccc library. 

### Requirements

  - CMake 3.25 or above
  - C++ compiler 20 or above
  - ACTS library
  - CUDA 12.6 or above
  - Boost 1.78 or above

### Workflow

- Simulation : set details in toy_mc.conf
  sh
  export TRACCC_TEST_DATA_DIR=$PWD/data
  mkdir -p data/run && cd data/run
  ../../traccc-build/bin/traccc_toy_mc ../../config/toy_mc.conf
  cd ../..

- Digitization : set details in toy_digitization.conf
  sh
  python3 src/digitization/toy_digitize.py data/run --config config/toy_digitization.conf

- CPU
  sh
  data=$(realpath data/run)

  ./traccc-build/bin/traccc_reco_cpu \
    --detector-file=$data/toy_detector_geometry.json \
    --material-file=$data/toy_detector_homogeneous_material.json \
    --grid-file=$data/toy_detector_surface_grids.json \
    --digitization-file=$data/toy_digitization.json \
    --conditions-file=$data/toy_digitization.json \
    --use-acts-geom-source=false \
    --input-directory=$data/ \
    --seedfinder-r-range=25:200 \
    --finding-run-smoother=none \
    --constraint-step-size-mm=1 \
    --search-window=3:3 \
    --input-events=NUMBER_OF_EVENTS

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

- Throughput 
  ./run_throughput.sh cuda data/run
  ./run_throughput.sh alpaka data/run --cpu-threads=8