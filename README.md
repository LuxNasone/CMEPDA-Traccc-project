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

- Simulation + Digitization : 

  - Set details :  simulation.conf
  - Compilation : cmake --build traccc-build -j8 --target traccc_toy_mc 2>&1 
  - Execution : ./exec/run_simulation.sh

- Reconstruction (substitute gpu with cpu for cpu execution):
  
  - Set details : reco.conf
  - Compilation : cmake --build traccc-build -j8 --target traccc_reco_gpu 2>&1
  - Execution :  ./exec/run_reco.sh gpu data/run_* --input_event (NUMEROEVENTI)
