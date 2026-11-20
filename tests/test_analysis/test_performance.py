'''
@brief Tests of src/analysis/performance.py, on small performance files and
       nvidia-smi dmon logs written in a temporary folder.
'''

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import timing_performance as perf

## The script, to run it as a program
SCRIPT = Path(perf.__file__)

## Columns of the performance files written by the reconstruction
STEP_COLUMNS = ["t_h2d_ms", "t_clusterization_ms", "t_spacepoints_ms", "t_seeding_ms",
                "t_params_ms", "t_finding_ms", "t_fitting_ms", "t_d2h_ms"]


def write_run(path, n_events=10, threads=None, spacepoints=100, step_ms=1.0, read_ms=5.0):
    '''
    @brief writes a performance file with constant times
    @param path : (Path) file to write
    @param n_events : (int) number of events
    @param threads : (int) number of threads; None = old format, without the
                     thread and time columns
    @param spacepoints : (int) spacepoints of every event
    @param step_ms : (float) time of every reconstruction step
    @param read_ms : (float) time of the file reading
    @return the DataFrame written
    '''
    data = pd.DataFrame({
        "backend": "cuda", "event": range(n_events),
        "n_cells": 2 * spacepoints, "n_measurements": spacepoints,
        "n_spacepoints": spacepoints, "n_seeds": spacepoints, "n_found_tracks": 10,
        "n_fitted_tracks": 10, "t_read_ms": read_ms, "t_wall_ms": 0.0,
    })
    for column in STEP_COLUMNS:
        data[column] = step_ms
    if threads is not None:
        data["n_threads"] = threads
        data["thread"] = np.arange(n_events) % threads
        data["t_start_ms"] = np.arange(n_events) * 10.0
        data["t_end_ms"] = data["t_start_ms"] + 10.0
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data


def write_dmon_log(path, sm_values, with_time=True):
    '''
    @brief writes a log in the format of "nvidia-smi dmon -s u"
    @param path : (Path) file to write
    @param sm_values : list of sm values, one per second (mem = sm / 2)
    @param with_time : (bool) True = with the time column (dmon -o T)
    '''
    if with_time:
        lines = ["#Time        gpu     sm    mem    enc    dec    jpg    ofa",
                 "#HH:MM:SS    Idx      %      %      %      %      %      %"]
    else:
        lines = ["# gpu     sm    mem    enc    dec    jpg    ofa",
                 "# Idx      %      %      %      %      %      %"]
    for second, sm in enumerate(sm_values):
        prefix = f"10:00:{second:02d}  " if with_time else ""
        lines.append(f"{prefix}    0     {sm}     {sm // 2}      0      0      -      -")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# --- names and warm-up -----------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("gpu_run_500.csv", 500), ("cpu_run_10.csv", 10), ("summary.csv", None),
    ("event000000003-tracks.csv", None)])
def test_run_number(name, expected):
    assert perf.run_number(Path(name)) == expected


def test_drop_warmup_old_format_drops_first_row():
    data = pd.DataFrame({"event": range(5)})
    assert list(perf.drop_warmup(data)["event"]) == [1, 2, 3, 4]


def test_drop_warmup_drops_first_event_of_every_thread():
    data = pd.DataFrame({"event": range(6), "thread": [0, 1, 2, 0, 1, 2],
                         "t_start_ms": [0, 1, 2, 3, 4, 5]})
    assert list(perf.drop_warmup(data)["event"]) == [3, 4, 5]


# --- summary of one run ----------------------------------------------------

def test_summarize_excludes_file_reading(tmp_path):
    path = tmp_path / "gpu_run_100.csv"
    write_run(path, n_events=10, spacepoints=300, step_ms=2.0, read_ms=50.0)
    row = perf.summarize(path)
    assert row["particles"] == 100
    assert row["events"] == 9                      # warm-up event removed
    assert row["n_spacepoints"] == 300
    assert row["t_reco_ms"] == pytest.approx(2.0 * len(STEP_COLUMNS))
    assert row["t_read_ms"] == pytest.approx(50.0)
    assert "events_per_s" not in row               # old format: no throughput


def test_summarize_throughput(tmp_path):
    path = tmp_path / "gpu_run_100.csv"
    write_run(path, n_events=10, threads=2)
    row = perf.summarize(path)
    # Events 2..9 remain (8 events), from t = 20 ms to t = 100 ms
    assert row["events"] == 8
    assert row["events_per_s"] == pytest.approx(1000.0 * 8 / 80.0)


def test_summarize_rejects_a_file_with_only_the_warmup(tmp_path):
    path = tmp_path / "gpu_run_10.csv"
    write_run(path, n_events=1)
    with pytest.raises(ValueError):
        perf.summarize(path)


def test_load_series_ignores_other_files_and_sorts(tmp_path):
    write_run(tmp_path / "gpu_run_500.csv", spacepoints=5000)
    write_run(tmp_path / "gpu_run_10.csv", spacepoints=100)
    (tmp_path / "summary.csv").write_text("a,b\n1,2\n")
    series = perf.load_series(tmp_path)
    assert list(series["particles"]) == [10, 500]


def test_load_series_without_runs_stops(tmp_path):
    with pytest.raises(SystemExit):
        perf.load_series(tmp_path)


# --- GPU utilization logs --------------------------------------------------

@pytest.mark.parametrize("with_time", [True, False])
def test_read_gpu_util_trims_idle_start_and_end(tmp_path, with_time):
    path = tmp_path / "run_10.log"
    write_dmon_log(path, [0, 0, 40, 0, 60, 0], with_time=with_time)
    data = perf.read_gpu_util(path)
    # Idle seconds kept only between busy ones
    assert list(data["sm"]) == [40, 0, 60]
    assert list(data["mem"]) == [20, 0, 30]


def test_read_gpu_util_drops_a_cut_line(tmp_path):
    path = tmp_path / "run_10.log"
    write_dmon_log(path, [50, 50])
    with open(path, "a") as log:
        log.write("10:00:09    0     7\n")       # line cut by kill
    assert len(perf.read_gpu_util(path)) == 2


def test_read_gpu_util_of_an_idle_gpu_is_empty(tmp_path):
    path = tmp_path / "run_10.log"
    write_dmon_log(path, [0, 0, 0])
    assert perf.read_gpu_util(path).empty


def test_load_gpu_util(tmp_path):
    write_dmon_log(tmp_path / "run_10.log", [0, 20, 40, 0])
    write_dmon_log(tmp_path / "run_50.log", [80])
    util = perf.load_gpu_util(tmp_path).set_index("particles")
    assert util.loc[10, "gpu_sm"] == pytest.approx(30.0)
    assert util.loc[50, "gpu_samples"] == 1


# --- folders of a thread scan ----------------------------------------------

def make_scan(root, threads=(1, 2, 16), with_logs=(2,)):
    '''
    @brief writes a thread scan: threads_N folders with two runs each and
           gpu_util_tN folders for some of them
    '''
    for n in threads:
        for particles in (10, 100):
            # More events than threads: the first event of every thread is dropped
            write_run(root / f"threads_{n}" / f"gpu_run_{particles}.csv",
                      n_events=2 * max(threads), threads=n)
    for n in with_logs:
        for particles in (10, 100):
            write_dmon_log(root / f"gpu_util_t{n}" / f"run_{particles}.log", [30, 50])


def test_expand_folders_natural_order(tmp_path):
    make_scan(tmp_path)
    folders = perf.expand_folders([str(tmp_path)])
    assert [f.name for f in folders] == ["threads_1", "threads_2", "threads_16"]


def test_series_label_and_util_folder(tmp_path):
    make_scan(tmp_path)
    assert perf.series_label(tmp_path / "threads_1") == "1 thread"
    assert perf.series_label(tmp_path / "threads_16") == "16 threads"
    assert perf.series_label(tmp_path / "gpu") == "gpu"
    assert perf.util_folder(tmp_path / "threads_2") == tmp_path / "gpu_util_t2"
    assert perf.util_folder(tmp_path / "threads_1") is None


# --- the whole program -----------------------------------------------------

def run_script(*args):
    '''
    @brief runs performance.py as a program, without a display
    @return the completed process
    '''
    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, env=env, check=False)


def test_program_writes_pdf_and_table(tmp_path):
    make_scan(tmp_path / "reco")
    result = run_script(tmp_path / "reco", "-o", tmp_path / "out")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "reco.pdf").stat().st_size > 0
    table = pd.read_csv(tmp_path / "out" / "reco.csv")
    assert set(table["series"]) == {"1 thread", "2 threads", "16 threads"}
    # GPU logs only for the 2-thread series
    has_util = table.groupby("series")["gpu_sm"].apply(lambda s: s.notna().any())
    assert has_util.to_dict() == {"1 thread": False, "16 threads": False, "2 threads": True}


def test_program_rejects_wrong_number_of_labels(tmp_path):
    make_scan(tmp_path / "reco")
    result = run_script(tmp_path / "reco", "-l", "only one", "-o", tmp_path / "out")
    assert result.returncode != 0
    assert "label" in result.stderr