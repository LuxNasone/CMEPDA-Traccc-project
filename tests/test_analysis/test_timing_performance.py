"""
Tests of the folder layouts read by timing_performance.py: one performance
file per run (gpu_run_<N>.csv) and the layout of run_reco.sh, with one folder
per run (run_<N>/logs/<backend>_performance.csv, run_<N>/logs/gpu_util.log,
run_<N>/hough/performance.csv).
"""
from pathlib import Path

import pandas as pd
import pytest

import timing_performance as perf

## Lines of a "nvidia-smi dmon -s u -o T" log: 3 busy seconds between idle ones
DMON_LOG = """\
#Time        gpu     sm    mem    enc    dec
#HH:MM:SS    Idx      %      %      %      %
12:00:00       0      0      0      0      0
12:00:01       0     40     20      0      0
12:00:02       0     60     30      0      0
12:00:03       0     50     10      0      0
12:00:04       0      0      0      0      0
"""


def write_traccc_performance(path, n_spacepoints, n_events=5, threads=1):
    '''
    @brief writes a performance file of traccc_reco with n_events events, all
           with the same times (1 ms per step), spread over the threads
    @param path : Path of the file (its folders are created)
    @param n_spacepoints : spacepoints of every event
    @param n_events : number of events
    @param threads : number of threads the events are spread over
    '''
    rows = []
    for event in range(n_events):
        row = {"backend": "cuda", "event": event, "thread": event % threads,
               "n_spacepoints": n_spacepoints, "t_read_ms": 2.0,
               "t_start_ms": 10.0 * event, "t_end_ms": 10.0 * event + 8.0}
        for step in ("t_h2d_ms", "t_clusterization_ms", "t_spacepoints_ms",
                     "t_seeding_ms", "t_params_ms", "t_finding_ms",
                     "t_fitting_ms", "t_d2h_ms"):
            row[step] = 1.0
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_hough_performance(path, n_spacepoints, n_events=5, threads=1):
    '''
    @brief writes a performance file of traccc_hough_reco (3 ms of Hough
           transform per event), with the columns of the multithreaded version
    @param path : Path of the file (its folders are created)
    @param n_spacepoints : spacepoints of every event
    @param n_events : number of events
    @param threads : number of threads the events are spread over
    '''
    rows = [{"backend": "hough", "event": e, "thread": e % threads,
             "n_spacepoints": n_spacepoints, "n_fitted_tracks": 10,
             "t_read_ms": 1.0, "t_hough_ms": 3.0, "t_wall_ms": 4.5,
             "t_start_ms": 5.0 * e, "t_end_ms": 5.0 * e + 4.5}
            for e in range(n_events)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def reco_tree(tmp_path):
    '''
    @brief results/reco as written by run_reco.sh and traccc_hough_reco: two
           thread counts, two runs each, with GPU logs and Hough results, plus
           a summary table of a previous analysis that must be ignored
    @return Path of the reco folder
    '''
    reco = tmp_path / "reco"
    for threads in (1, 2):
        series = reco / f"threads_{threads}"
        for particles, n_sp in ((10, 50), (100, 500)):
            run = series / f"run_{particles}"
            write_traccc_performance(run / "logs" / "gpu_performance.csv", n_sp,
                                     threads=threads)
            (run / "logs" / "gpu_util.log").write_text(DMON_LOG)
            (run / "logs" / "gpu_reco.log").write_text("output of traccc 123\n")
            write_hough_performance(run / "hough" / "performance.csv", n_sp,
                                    threads=threads)
        # Output of a previous run of the script: not a run
        (series / f"threads_{threads}.csv").write_text("series,run\n")
    return reco


# --- names of the runs ------------------------------------------------------

@pytest.mark.parametrize("path, name, number", [
    ("results/reco/gpu_run_500.csv", "gpu_run_500", 500),
    ("results/reco/gpu_util_t1/run_500.log", "run_500", 500),
    ("reco/threads_8/run_500/logs/gpu_performance.csv", "run_500", 500),
    ("reco/threads_8/run_500/logs/cpu_performance.csv", "run_500", 500),
    ("reco/threads_8/run_500/logs/gpu_util.log", "run_500", 500),
    ("reco/threads_8/run_500/hough/performance.csv", "run_500", 500),
])
def test_run_name_and_number(path, name, number):
    '''
    @brief the run is found from the file name or from its run_<N> folder
    '''
    assert perf.run_name(Path(path)) == name
    assert perf.run_number(Path(path)) == number


@pytest.mark.parametrize("path", [
    "reco/threads_8/threads_8.csv",          # summary table of this script
    "reco/reco.csv",
    "reco/threads_8/run_500/gpu_performance.csv",   # not in logs/
    "reco/threads_8/other/logs/gpu_performance.csv",  # folder is not run_<N>
    "reco/threads_8/run_500/tracks/event000000000-tracks.csv",
])
def test_not_a_run(path):
    '''
    @brief files that do not belong to a run give None, and no error
    '''
    assert perf.run_name(Path(path)) is None
    assert perf.run_number(Path(path)) is None


# --- files of a series ------------------------------------------------------

def test_performance_files_per_backend(reco_tree):
    '''
    @brief every backend finds only its own files; the summary table and the
           logs of traccc are never taken
    '''
    series = reco_tree / "threads_1"
    gpu = perf.performance_files(series, "gpu")
    hough = perf.performance_files(series, "hough")
    assert sorted(p.parent.parent.name for p in gpu) == ["run_10", "run_100"]
    assert all(p.name == "gpu_performance.csv" for p in gpu)
    assert sorted(p.parent.parent.name for p in hough) == ["run_10", "run_100"]
    assert all(p.parent.name == "hough" for p in hough)
    assert perf.performance_files(series, "cpu") == []


def test_performance_files_flat_layout(tmp_path):
    '''
    @brief the old layout, one <backend>_run_<N>.csv per run, is still read
    '''
    write_traccc_performance(tmp_path / "gpu_run_10.csv", 50)
    write_traccc_performance(tmp_path / "gpu_run_100.csv", 500)
    (tmp_path / "notes.csv").write_text("a,b\n")
    files = perf.performance_files(tmp_path)
    assert sorted(p.name for p in files) == ["gpu_run_10.csv", "gpu_run_100.csv"]


def test_expand_folders_finds_every_thread_count(reco_tree):
    '''
    @brief the reco folder gives one series per threads_N, in natural order
    '''
    folders = perf.expand_folders([str(reco_tree)])
    assert [f.name for f in folders] == ["threads_1", "threads_2"]
    assert [f.name for f in perf.expand_folders([str(reco_tree)], "hough")] == \
        ["threads_1", "threads_2"]


def test_expand_folders_without_runs(tmp_path):
    '''
    @brief a folder without performance files stops the script with a message
    '''
    with pytest.raises(SystemExit):
        perf.expand_folders([str(tmp_path)])
    with pytest.raises(SystemExit):
        perf.expand_folders([str(tmp_path)], "hough")


# --- summaries --------------------------------------------------------------

def test_load_series_traccc(reco_tree):
    '''
    @brief one row per run, sorted by spacepoints, warm-up event of every
           thread removed; the reconstruction time is the sum of the 8 steps
    '''
    s = perf.load_series(reco_tree / "threads_2", "gpu")
    assert list(s["run"]) == ["run_10", "run_100"]
    assert list(s["particles"]) == [10, 100]
    assert list(s["events"]) == [3, 3]          # 5 events - 1 per thread
    assert s["t_reco_ms"].tolist() == pytest.approx([8.0, 8.0])
    assert s["events_per_s"].notna().all()


def test_load_series_hough(reco_tree):
    '''
    @brief the Hough series uses t_hough_ms as reconstruction time, and has a
           throughput thanks to the thread and t_start/t_end columns
    '''
    s = perf.load_series(reco_tree / "threads_1", "hough")
    assert list(s["particles"]) == [10, 100]
    assert s["t_reco_ms"].tolist() == pytest.approx([3.0, 3.0])
    assert s["t_hough_ms"].tolist() == pytest.approx([3.0, 3.0])
    # 4 events (warm-up removed) from t = 5 to t = 24.5 ms
    assert s["events_per_s"].tolist() == pytest.approx([1000. * 4 / 19.5] * 2)


# --- GPU utilization --------------------------------------------------------

def test_util_folder_new_layout(reco_tree):
    '''
    @brief with the run_reco.sh layout the GPU logs are in the series itself
    '''
    series = reco_tree / "threads_1"
    assert perf.util_folder(series) == series


def test_util_folder_old_layout(tmp_path):
    '''
    @brief with the old layout the logs of threads_N are in gpu_util_tN
    '''
    (tmp_path / "threads_4").mkdir()
    (tmp_path / "gpu_util_t4").mkdir()
    assert perf.util_folder(tmp_path / "threads_4") == tmp_path / "gpu_util_t4"
    (tmp_path / "threads_8").mkdir()
    assert perf.util_folder(tmp_path / "threads_8") is None


def test_load_gpu_util_new_layout(reco_tree):
    '''
    @brief only the gpu_util.log of every run are read (not the logs of
           traccc), and the idle seconds at the edges are removed
    '''
    util = perf.load_gpu_util(reco_tree / "threads_1").sort_values("particles")
    assert list(util["particles"]) == [10, 100]
    assert util["gpu_sm"].tolist() == pytest.approx([50.0, 50.0])
    assert util["gpu_mem"].tolist() == pytest.approx([20.0, 20.0])
    assert list(util["gpu_samples"]) == [3, 3]