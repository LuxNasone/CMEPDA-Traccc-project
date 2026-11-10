#!/usr/bin/env bash
# Runs the toy MC simulation and then the digitization of its events, with the
# options of one configuration file (default: config/simulation.conf).
#
# Usage: ./run_simulation.sh [config file] [key=value ...]
# Examples:
#   ./run_simulation.sh
#   ./run_simulation.sh config/simulation.conf gen-nparticles=500 output-directory=data/run_500
#   pytest -q tests/ && for n in 10 50 100 500; do
#       SKIP_TESTS=1 ./run_simulation.sh config/simulation.conf gen-nparticles=$n output-directory=data/run_$n
#   done
#
# A key=value given on the command line replaces the value of that key in the
# configuration file. The output folder is data/<output-directory>.
# Before the simulation the tests of the digitization (tests/) are run; set
# SKIP_TESTS=1 to skip them, e.g. inside a loop after the first run.
set -euo pipefail

config=config/simulation.conf
if [ $# -gt 0 ] && [[ "$1" != *=* ]]; then
    config=$1
    shift
fi
overrides=("$@")

toy_mc=$PWD/traccc-build/bin/traccc_toy_mc
digitizer=$PWD/src/digitization/toy_digitize.py

if [ ! -f "$config" ]; then
    echo "Configuration file not found: $config" >&2
    exit 1
fi
if [ ! -x "$toy_mc" ]; then
    echo "Application not found: $toy_mc (run this script from the project folder)" >&2
    exit 1
fi

# Tests of the digitization: if one fails, nothing is simulated.
if [ "${SKIP_TESTS:-0}" != 1 ]; then
    echo "==> Tests of the digitization"
    pytest -q tests/
fi

# Read the configuration as "section<TAB>key<TAB>value" lines.
entries=$(awk '
    { sub(/#.*/, ""); gsub(/^[ \t]+|[ \t\r]+$/, "") }
    $0 == "" { next }
    /^\[.*\]$/ { section = substr($0, 2, length($0) - 2); next }
    {
        eq = index($0, "=")
        if (eq == 0 || section == "") {
            printf "Invalid line (outside a section or without =): %s\n", $0 > "/dev/stderr"
            exit 1
        }
        key = substr($0, 1, eq - 1); value = substr($0, eq + 1)
        gsub(/[ \t]+$/, "", key); gsub(/^[ \t]+/, "", value)
        print section "\t" key "\t" value
    }' "$config")

# Apply the command line overrides: each key must exist in the file.
for override in "${overrides[@]+"${overrides[@]}"}"; do
    if [[ "$override" != *=* ]]; then
        echo "Invalid option (expected key=value): $override" >&2
        exit 1
    fi
    key=${override%%=*}
    value=${override#*=}
    if ! printf '%s\n' "$entries" | cut -f2 | grep -qxF "$key"; then
        echo "Unknown key: $key (it is not in $config)" >&2
        exit 1
    fi
    entries=$(printf '%s\n' "$entries" |
        awk -F'\t' -v OFS='\t' -v k="$key" -v v="$value" '$2 == k { $3 = v } { print }')
done

# Value of a key of the [common] section.
common() {
    printf '%s\n' "$entries" | awk -F'\t' -v k="$1" '$1 == "common" && $2 == k { print $3 }'
}
seed=$(common seed)
out_dir=$(common output-directory)
if [ -z "$seed" ] || [ -z "$out_dir" ]; then
    echo "The [common] section must contain seed and output-directory" >&2
    exit 1
fi

data_root=$PWD/data
out=$data_root/$out_dir
mkdir -p "$out"

# Configuration files of the two steps, in a temporary folder removed at the end.
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
toy_conf=$tmp/toy_mc.conf
digi_conf=$tmp/toy_digitization.conf
{
    echo "# Written by run_simulation.sh from $config"
    printf '%s\n' "$entries" | awk -F'\t' '$1 == "toy_mc" { print $2 " = " $3 }'
    echo "seed = $seed"
    echo "output-directory = $out_dir"
} > "$toy_conf"
{
    echo "# Written by run_simulation.sh from $config"
    printf '%s\n' "$entries" | awk -F'\t' '$1 == "digitization" { print $2 " = " $3 }'
    echo "seed = $seed"
} > "$digi_conf"

echo "==> Simulation: $out"
# The toy MC writes the events in TRACCC_TEST_DATA_DIR/<output-directory> and
# the detector files in the current folder, so it runs inside the output folder.
# Its configuration file is read at start, so the temporary folder can be used.
(cd "$out" && TRACCC_TEST_DATA_DIR=$data_root "$toy_mc" "$toy_conf")

echo "==> Digitization: $out"
mkdir -p logs
python3 "$digitizer" "$out" --config "$digi_conf" \
    --log-file "logs/toy_digitize_$(basename "$out").log"

echo "Done: events, detector files and cells in $out"

