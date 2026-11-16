#!/usr/bin/env bash
# Reconstruction of every run with 1..8 threads, recording the GPU utilization.
cd ~/Exam_2
for t in 1 2 3 4 5 6 7 8; do
    out=results/reco/gpu_util_t$t
    mkdir -p "$out"
    for run in data/run_100ev/run_*; do
        name=$(basename "$run")
        echo "### threads=$t  $name  $(date +%H:%M:%S)"
        nvidia-smi dmon -s u -i 0 -o T > "$out/$name.log" &
        dmon_pid=$!
        ./exec/run_reco.sh gpu "$run" -- --input-events=100 --threads=$t
        kill $dmon_pid 2>/dev/null
        wait $dmon_pid 2>/dev/null
    done
done
echo "### finished $(date +%H:%M:%S)"
