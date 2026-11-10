#!/usr/bin/env bash
# Runs traccc_reco_<cpu|gpu> on one or more data folders. For each folder it
# saves the log and a copy of the per-event performance file in results/reco/,
# because the program always writes to the same output folder.
#
# Usage: ./run_reco.sh <cpu|gpu> <data folder>... [-- extra traccc options]
# Examples:
#   ./run_reco.sh gpu data/data/run_10
#   ./run_reco.sh gpu data/data/run_*
#   ./run_reco.sh gpu data/data/run_* -- --input-events=5
#
# The traccc options are read from config/reco.conf (or from the file in the
# RECO_CONFIG variable). RECO_OUT is the folder the program writes to, i.e.
# RECO_DATA_DIR in CMake (default: data).
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

app=traccc-build/bin/traccc_reco_${backend}
config=${RECO_CONFIG:-config/reco.conf}
reco_out=${RECO_OUT:-data}/reco_${backend}
results=results/reco

if [ ! -x "$app" ]; then
    echo "Application not found: $app (run this script from the project folder)" >&2
    exit 1
fi
if [ ! -f "$config" ]; then
    echo "Configuration file not found: $config" >&2
    exit 1
fi
mkdir -p "$results"

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
        "${conf_options[@]}"
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

    log="$results/${backend}_${dataset}.log"
    echo "==> $backend on $dataset"
    # Remove the file of the previous dataset, so that it is never copied by mistake.
    rm -f "$reco_out/performance.csv"
    # A failure on one dataset must not stop the others.
    if ! "$app" "${options[@]}" "${extra[@]+"${extra[@]}"}" 2>&1 | tee "$log"; then
        echo "FAILED: $backend on $dataset, see $log" >&2
        failed+=("$dataset")
        continue
    fi

    if [ -f "$reco_out/performance.csv" ]; then
        cp "$reco_out/performance.csv" "$results/${backend}_${dataset}.csv"
        echo "Saved: $results/${backend}_${dataset}.csv and $log"
    else
        echo "No $reco_out/performance.csv found: set RECO_OUT to the folder" \
             "that contains reco_${backend}" >&2
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed datasets: ${failed[*]} (see the logs in $results)" >&2
    exit 1
fi

