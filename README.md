# CMEPDA Traccc performance project

## Brief description

This project measures the performance of the [traccc](https://github.com/acts-project/traccc)
track reconstruction library on GPU and CPU. A toy Monte Carlo simulates the
events of a detector, a Python digitizer turns them into pixel cells, and a
reconstruction program runs the full traccc chain (clusterization, spacepoint
formation, seeding, track finding, track fitting) on each event.

## Requirements

- Linux, `git` and an internet connection (ACTS is downloaded by the build)
- CMake 3.25 or above
- C++20 compiler (e.g. GCC 11 or above)
- CUDA 12.6 or above, for the GPU programs (optional: without it only the CPU
  programs are built)
- Boost 1.78 or above
- Python 3 with the packages of `requirements.txt`:
  ```bash
  python3 -m venv venv && source venv/bin/activate
  pip install -r requirements.txt
  ```

ACTS, traccc and Alpaka do not need to be installed: `exec/build.sh` takes care of them.

## Quick start

All commands are run from the project folder.

```bash
# 1. Build (the first time it downloads ACTS and compiles everything)
./exec/build.sh

# 2. Simulate and digitize the events
./exec/run_simulation.sh config/simulation.conf gen-nparticles=500 output-directory=run_500

# 3. Reconstruct them, on the GPU with 4 threads
./exec/run_reco.sh gpu data/run_500 -- --input-events=100 --threads=4

# 4. Plot the performance
python3 src/analysis/performance.py results/reco/threads_4
```

## Workflow

### 1. Build — `exec/build.sh`

Starting from a fresh clone, the script

1. downloads ACTS (which contains traccc) at the commit the project was developed with;
2. applies the project changes to ACTS, stored in `patches/acts_local_changes.patch`
   (targets of the reconstruction and simulation programs);
3. configures traccc in `traccc-build/` (only the first time);
4. compiles `traccc_toy_mc`, `traccc_reco_cpu` and, if CUDA is found, `traccc_reco_gpu`.

It can be run again at any time, e.g. after changing the code: only what changed is compiled.
Settings, through environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `CUDA_ROOT` | CUDA installation | `/usr/local/cuda-13.0` |
| `BOOST_ROOT` | Boost installation | `$HOME/local` |
| `BUILD_DIR` | build folder | `traccc-build` |
| `JOBS` | parallel compilation jobs | number of cores |
| `RECONFIGURE` | `1` = run the CMake configuration again | `0` |

Example: `CUDA_ROOT=/usr/local/cuda-12.6 JOBS=4 ./exec/build.sh`

`prebuilt-codegen/` holds code that ACTS would otherwise generate with Python
during the build; it avoids extra Python dependencies on the build machine.

### 2. Simulation and digitization — `exec/run_simulation.sh`

```bash
./exec/run_simulation.sh [config file] [key=value ...]
```

The options are read from `config/simulation.conf`; a `key=value` on the command
line replaces the value of that key. The events, the detector files and the
cells are written to `data/<output-directory>/`. Before the simulation the
tests of the digitization are run (`SKIP_TESTS=1` skips them).

A scan over the number of particles per event:

```bash
pytest -q tests/
for n in 10 50 100 500 1000 5000 10000; do
    SKIP_TESTS=1 ./exec/run_simulation.sh config/simulation.conf \
        gen-nparticles=$n output-directory=run_100ev/run_$n
done
```

### 3. Reconstruction — `exec/run_reco.sh`

```bash
./exec/run_reco.sh <cpu|gpu> <data folder>... [-- extra options]
```

Runs `traccc_reco_cpu` or `traccc_reco_gpu` on one or more data folders, with the
options of `config/reco.conf`. Options after `--` are passed to the program and
replace those of the configuration file. The most useful ones:

| Option | Meaning |
| --- | --- |
| `--input-events=N` | number of events to reconstruct |
| `--threads=N` | events processed in parallel, each thread with its own CUDA stream (default 1) |

```bash
./exec/run_reco.sh gpu data/run_100ev/run_* -- --input-events=100
./exec/run_reco.sh gpu data/run_100ev/run_* -- --input-events=100 --threads=4
./exec/run_reco.sh cpu data/run_100ev/run_* -- --input-events=100
```

For each data folder the script saves, in `results/reco/` (or
`results/reco/threads_N/` with `--threads=N`):

- `<backend>_<run>.log`: the output of the program, ending with the throughput
  (`Event loop: … events/s`);
- `<backend>_<run>.csv`: one row per event with its size (cells, measurements,
  spacepoints, seeds, tracks), the time of each reconstruction step, the thread
  that processed it and its start and end time.

The reconstructed tracks of each event are written to `data/reco_<cpu|gpu>/`.

### 4. Analysis — `src/analysis/performance.py`

```bash
python3 src/analysis/performance.py <results folder>... [-l labels...] [-u GPU logs...] [-n name]
```

Each results folder is one series of the plots, e.g. GPU with 1 and 4 threads:

```bash
python3 src/analysis/performance.py results/reco results/reco/threads_4 \
    -l "1 thread" "4 threads" -n gpu_threads
```

The plots are saved in `results/<name>.pdf`, with a table of the numbers in
`results/<name>.csv`:

- reconstruction time per event (file reading excluded) against the number of spacepoints;
- time of each reconstruction step, for every run;
- throughput (events per second);
- GPU utilization, if the logs of `nvidia-smi dmon` are given with `-u`.

To record the GPU utilization during the reconstruction of every run:

```bash
mkdir -p results/reco/gpu_util_t4
for run in data/run_100ev/run_*; do
    nvidia-smi dmon -s u -i 0 -o T > results/reco/gpu_util_t4/$(basename $run).log &
    dmon_pid=$!
    ./exec/run_reco.sh gpu "$run" -- --input-events=100 --threads=4
    kill $dmon_pid
done
```

## Tests

```bash
pytest -q tests/
```

## Repository structure

```
config/              options of the simulation and of the reconstruction
exec/                build.sh, run_simulation.sh, run_reco.sh
patches/             project changes to ACTS, applied by build.sh
prebuilt-codegen/    pregenerated ACTS code, used by build.sh
src/
  simulation/        toy Monte Carlo (C++)
  digitization/      digitizer: simulated hits -> pixel cells (Python)
  reconstruction/    reconstruction chain on CPU and GPU (C++)
  analysis/          performance plots (Python)
  python_interface/  reading of the traccc files in Python
tests/               tests of the Python code
```

Generated folders, not in the repository: `acts/`, `traccc-build/`, `data/`,
`results/`, `venv/`.