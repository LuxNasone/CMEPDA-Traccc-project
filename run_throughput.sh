#!/usr/bin/env bash
# Runs a traccc throughput application on a data folder, saves the full log and
# appends the measured throughput to results/throughput/summary.csv.
#
# Usage: ./run_throughput.sh <cuda|alpaka> <data folder> [extra traccc options]
# Example: ./run_throughput.sh cuda data/single_track --cpu-threads=4 --processed-events=2000
# Extra options must be written as --name=value.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cuda|alpaka> <data folder> [extra traccc options]" >&2
    exit 1
fi

backend=$1
data=$(realpath "$2")
shift 2

app=traccc-build/bin/traccc_throughput_mt_${backend}
if [ ! -x "$app" ]; then
    echo "Application not found: $app" >&2
    exit 1
fi
if [ ! -d "$data" ]; then
    echo "Data folder not found: $data" >&2
    exit 1
fi

out=results/throughput
mkdir -p "$out"
stamp=$(date +%Y%m%d_%H%M%S)_$$
dataset=$(basename "$data")
log="$out/${stamp}_${backend}_${dataset}.log"

# Default options. An option given on the command line replaces its default:
# traccc refuses options that appear twice.
defaults=(
    "--detector-file=$data/toy_detector_geometry.json"
    "--material-file=$data/toy_detector_homogeneous_material.json"
    "--grid-file=$data/toy_detector_surface_grids.json"
    "--digitization-file=$data/toy_digitization.json"
    "--conditions-file=$data/toy_digitization.json"
    "--use-acts-geom-source=false"
    "--input-directory=$data/"
    "--seedfinder-r-range=25:200"
    "--finding-run-smoother=none"
    "--constraint-step-size-mm=1"
    "--search-window=3:3"
    "--processed-events=1000"
    "--cold-run-events=100"
    "--cpu-threads=1"
)

options=()
for default in "${defaults[@]}"; do
    key=${default%%=*}
    overridden=false
    for arg in "$@"; do
        if [ "${arg%%=*}" = "$key" ]; then
            overridden=true
        fi
    done
    if [ "$overridden" = false ]; then
        options+=("$default")
    fi
done

"$app" "${options[@]}" "$@" 2>&1 | tee "$log"

# Last "Event processing" line: "... <ms> ms/event, <rate> events/s"
line=$(grep "Event processing" "$log" | grep "ms/event" | tail -n 1)
ms_per_event=$(echo "$line" | sed -E 's/.* ([0-9.eE+-]+) ms\/event.*/\1/')
events_per_s=$(echo "$line" | sed -E 's/.*, ([0-9.eE+-]+) events\/s.*/\1/')

summary="$out/summary.csv"
if [ ! -f "$summary" ]; then
    echo "timestamp,host,backend,dataset,ms_per_event,events_per_s,extra_options,log" > "$summary"
fi
echo "$stamp,$(hostname),$backend,$dataset,$ms_per_event,$events_per_s,\"$*\",$log" >> "$summary"

echo
echo "Saved log: $log"
echo "Result:    $ms_per_event ms/event, $events_per_s events/s (added to $summary)"