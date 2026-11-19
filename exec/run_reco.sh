#!/usr/bin/env bash
# Runs traccc_reco_<cpu|gpu> on one or more data folders. The program always
# writes to the same folder (RECO_OUT/reco_<backend>), so after each dataset
# its files are moved to results/reco/threads_N/<dataset>/:
#   logs/<backend>_reco.log         output of the program
#   logs/<backend>_performance.csv  per-event times
#   logs/gpu_util.log               GPU utilization (gpu backend only)
#   tracks/                         event*-tracks.csv, event*-track-measurements.csv
#   spacepoints/                    event*-spacepoints.csv (with --write-spacepoints)
# The temporary folder RECO_OUT/reco_<backend> is removed at the end.
#
# Usage: ./exec/run_reco.sh <cpu|gpu> <data folder>... [-- extra traccc options]
# Examples:
#   ./exec/run_reco.sh gpu data/data/run_10
#   ./exec/run_reco.sh gpu data/data/run_*
#   ./exec/run_reco.sh gpu data/data/run_* -- --input-events=5
#   ./exec/run_reco.sh gpu data/data/run_* -- --input-events=100 --threads=4
#
# N is the value of --threads (1 if not given), so that runs with a different
# number of threads do not overwrite each other.
#
# With the gpu backend the GPU utilization is sampled every second with
# "nvidia-smi dmon" while each dataset is reconstructed. GPU_ID selects the
# GPU to monitor (default: 0).
#
# The traccc options are read from config/reco.conf (or from the file in the
# RECO_CONFIG variable). RECO_OUT/reco_<backend> is the folder the program
# writes to, i.e. RECO_DATA_DIR in CMake (default: results).
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cpu|gpu> <data folder>... [-- extra traccc options]" >&2
    exit 1
fi

backend=$1
shift
case "$backend" in
    cpu | gpu) ;;
    *) echo "Backend must be cpu or gpu, not '$backend'" >&2; exit 1 ;;
esac

# Split the arguments into data folders and extra options (after "--").
folders=()
extra=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then
        shift
        extra=("$@")
        break
    fi
    folders+=("$1")
    shift
done

# Number of threads (1 by default, as in traccc_reco), to keep the results of
# different runs apart.
threads_dir="threads_1"
for arg in "${extra[@]+"${extra[@]}"}"; do
    case "$arg" in
        --threads=*) threads_dir="threads_${arg#--threads=}" ;;
    esac
done

app=traccc-build/bin/traccc_reco_${backend}
config=${RECO_CONFIG:-config/reco.conf}
reco_out=${RECO_OUT:-results}/reco_${backend}
results=results/reco/$threads_dir
gpu_id=${GPU_ID:-0}

if [ ! -x "$app" ]; then
    echo "Application not found: $app (run this script from the project folder)" >&2
    exit 1
fi
if [ ! -f "$config" ]; then
    echo "Configuration file not found: $config" >&2
    exit 1
fi
mkdir -p "$results"

# GPU monitoring: only for the gpu backend, and only if nvidia-smi exists.
monitor_gpu=false
if [ "$backend" = gpu ]; then
    if command -v nvidia-smi > /dev/null; then
        monitor_gpu=true
    else
        echo "WARNING: nvidia-smi not found, the GPU utilization is not recorded" >&2
    fi
fi

# PID of the running "nvidia-smi dmon", empty when none is running.
dmon_pid=""
stop_gpu_monitor() {
    if [ -n "$dmon_pid" ]; then
        kill "$dmon_pid" 2> /dev/null || true
        wait "$dmon_pid" 2> /dev/null || true
        dmon_pid=""
    fi
}
# Stop the monitor also if the script ends early (error or Ctrl+C).
trap stop_gpu_monitor EXIT

# Options of the configuration file, without comments and empty lines.
conf_options=()
while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}
    line=$(echo "$line" | xargs)
    if [ -n "$line" ]; then
        conf_options+=("$line")
    fi
done < "$config"

failed=()
for folder in "${folders[@]}"; do
    if [ ! -d "$folder" ]; then
        echo "Data folder not found: $folder (skipped)" >&2
        continue
    fi
    data=$(realpath "$folder")
    dataset=$(basename "$data")

    defaults=(
        "--detector-file=$data/toy_detector_geometry.json"
        "--material-file=$data/toy_detector_homogeneous_material.json"
        "--grid-file=$data/toy_detector_surface_grids.json"
        "--digitization-file=$data/toy_digitization.json"
        "--conditions-file=$data/toy_digitization.json"
        "--input-directory=$data/"
        "${conf_options[@]+"${conf_options[@]}"}"
    )

    # An option given after "--" replaces the default with the same name:
    # traccc refuses options that appear twice.
    options=()
    for default in "${defaults[@]}"; do
        key=${default%%=*}
        overridden=false
        for arg in "${extra[@]+"${extra[@]}"}"; do
            if [ "${arg%%=*}" = "$key" ]; then
                overridden=true
            fi
        done
        if [ "$overridden" = false ]; then
            options+=("$default")
        fi
    done

    run_dir="$results/$dataset"
    logs_dir="$run_dir/logs"
    tracks_dir="$run_dir/tracks"
    sp_dir="$run_dir/spacepoints"
    log="$logs_dir/${backend}_reco.log"
    perf="$logs_dir/${backend}_performance.csv"
    util_log="$logs_dir/gpu_util.log"
    mkdir -p "$logs_dir"
    echo "==> $backend on $dataset ($threads_dir)"
    # Remove the files of the previous dataset, so that they are never moved
    # to the wrong folder.
    rm -f "$reco_out/performance.csv" "$reco_out"/event*.csv

    # GPU utilization, one sample per second, for this dataset only.
    if [ "$monitor_gpu" = true ]; then
        nvidia-smi dmon -s u -i "$gpu_id" -o T > "$util_log" &
        dmon_pid=$!
    fi

    # A failure on one dataset must not stop the others.
    status=0
    "$app" "${options[@]}" "${extra[@]+"${extra[@]}"}" 2>&1 | tee "$log" || status=$?
    stop_gpu_monitor

    if [ "$status" -ne 0 ]; then
        echo "FAILED: $backend on $dataset, see $log" >&2
        failed+=("$dataset")
        continue
    fi

    if [ -f "$reco_out/performance.csv" ]; then
        mv "$reco_out/performance.csv" "$perf"
    else
        echo "WARNING: no performance.csv in $reco_out" >&2
    fi

    # Tracks and spacepoints of this dataset, each in its own folder (the
    # folders of a previous run of the same dataset are replaced).
    rm -rf "$tracks_dir" "$sp_dir"
    shopt -s nullglob
    track_files=("$reco_out"/event*-track*.csv)
    sp_files=("$reco_out"/event*-spacepoints.csv)
    shopt -u nullglob
    if [ ${#track_files[@]} -gt 0 ]; then
        mkdir -p "$tracks_dir"
        mv "${track_files[@]}" "$tracks_dir/"
    fi
    if [ ${#sp_files[@]} -gt 0 ]; then
        mkdir -p "$sp_dir"
        mv "${sp_files[@]}" "$sp_dir/"
    fi

    echo "Saved in $run_dir"
done

# The temporary output folder of the program, if nothing is left in it.
rmdir "$reco_out" 2> /dev/null || true

if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed datasets: ${failed[*]} (see the logs in $results)" >&2
    exit 1
fi