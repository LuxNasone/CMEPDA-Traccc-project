import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from tqdm import tqdm
except ImportError:  # tqdm not installed: a minimal text progress bar
    import sys

    def tqdm(iterable, desc="", total=None, **_):
        '''
        @brief minimal replacement of tqdm: prints "desc [#####.....] i/n" on stderr
        @param iterable : items to loop over
        @param desc : (str) text before the bar
        @param total : (int) number of items, default len(iterable)
        @return the items of iterable, one at a time
        '''
        items = list(iterable)
        total = total or len(items)
        width = 30
        for i, item in enumerate(items, start=1):
            yield item
            done = int(width * i / total) if total else width
            sys.stderr.write(f"\r{desc} [{'#' * done}{'.' * (width - done)}] {i}/{total}")
            sys.stderr.flush()
        if total:
            sys.stderr.write("\n")

## Reconstruction steps, in chain order, with their labels. File reading is
## kept apart: it is the same for every algorithm and depends on the disk.
STEPS = {
    "t_h2d_ms": "H2D copy",
    "t_clusterization_ms": "Clusterization",
    "t_spacepoints_ms": "Spacepoint formation",
    "t_seeding_ms": "Seeding",
    "t_params_ms": "Track params",
    "t_finding_ms": "Track finding",
    "t_fitting_ms": "Track fitting",
    "t_d2h_ms": "D2H copy",
    "t_hough_ms": "Hough transform",   # hough_reco: replaces seeding to fitting
}


def save_plots(figs, output):
    '''
    @brief saves the figures in a single PDF file, one per page, overwriting it if it exists
    @param figs : list of matplotlib figures to save, in page order
    @param output : (str or Path) output PDF file
    '''
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)   
    with PdfPages(output) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    print(f"{len(figs)} plots saved in {output}")


def run_name(path):
    '''
    @brief name of the run a file belongs to. Two layouts are accepted:
           - one file per run, named after it: gpu_run_500.csv -> gpu_run_500
           - the layout of run_reco.sh, one folder per run:
             run_500/logs/gpu_performance.csv -> run_500, and for the Hough
             transform run_500/hough/performance.csv -> run_500
    @param path : Path of the performance file or of the GPU log
    @return (str) the name of the run, or None if the file belongs to none
    '''
    # Only names ending in run_<N>: the summary tables written by this script
    # (e.g. threads_8.csv) must not be taken for runs
    if re.search(r"run_\d+$", path.stem):
        return path.stem
    if path.parent.name in ("logs", "hough") and \
            re.fullmatch(r"run_\d+", path.parent.parent.name):
        return path.parent.parent.name
    return None


def run_number(path):
    '''
    @brief number of particles of a run, read from its name
           (gpu_run_500.csv or run_500/logs/gpu_performance.csv -> 500)
    @param path : Path of the performance file or of the GPU log
    @return (int) the number in the name, or None if there is none
    '''
    name = run_name(path)
    match = re.search(r"(\d+)$", name) if name else None
    return int(match.group(1)) if match else None


def performance_files(folder, backend="gpu"):
    '''
    @brief performance files of the runs of a series
    @param folder : (str or Path) folder of the series: either with one
           <backend>_run_<N>.csv per run, or with one run_<N>/ folder per run
           (layout of run_reco.sh: run_<N>/logs/<backend>_performance.csv, and
           run_<N>/hough/performance.csv for the Hough transform)
    @param backend : gpu or cpu (traccc_reco), or hough (traccc_hough_reco)
    @return list of Paths
    '''
    folder = Path(folder)
    if backend == "hough":
        return [p for p in folder.glob("run_*/hough/performance.csv")
                if run_number(p) is not None]
    flat = folder.glob("*.csv")
    nested = folder.glob(f"run_*/logs/{backend}_performance.csv")
    return [p for p in (*flat, *nested) if run_number(p) is not None]


def drop_warmup(data):
    '''
    @brief removes the warm-up events: the first event of every thread pays
           the start-up costs (CUDA context, caches) and is not representative
    @param data : DataFrame of one performance file
    @return the DataFrame without the warm-up events
    '''
    if {"thread", "t_start_ms"}.issubset(data.columns):
        first = data.groupby("thread")["t_start_ms"].idxmin()
        return data.drop(index=first)
    return data.iloc[1:]   

def summarize(path):
    '''
    @brief reads one performance file and summarizes its events
    @param path : Path of the performance file
    @return (dict) particles, median spacepoints, median and 16-84% band of
            the reconstruction time, median time of every step, throughput
    '''
    data = drop_warmup(pd.read_csv(path))
    if data.empty:
        raise ValueError(f"{path}: no events left after removing the warm-up")

    steps = [c for c in STEPS if c in data.columns]
    t_reco = data[steps].sum(axis=1)

    row = {
        "run": run_name(path),
        "particles": run_number(path),
        "events": len(data),
        "n_spacepoints": data["n_spacepoints"].median(),
        "t_reco_ms": t_reco.median(),
        "t_reco_lo_ms": t_reco.quantile(0.16),
        "t_reco_hi_ms": t_reco.quantile(0.84),
        "t_read_ms": data["t_read_ms"].median(),
    }
    for step in steps:
        row[step] = data[step].median()

    if {"t_start_ms", "t_end_ms"}.issubset(data.columns):
        span = data["t_end_ms"].max() - data["t_start_ms"].min()
        row["events_per_s"] = 1000. * len(data) / span if span > 0 else np.nan
    return row


def load_series(folder, backend="gpu"):
    '''
    @brief summarizes every performance file of a series, one row per run
    @param folder : (str or Path) folder of the series (see performance_files)
    @param backend : gpu or cpu (traccc_reco), or hough (traccc_hough_reco)
    @return DataFrame sorted by number of spacepoints
    '''
    files = performance_files(folder, backend)
    if not files:
        raise SystemExit(f"No performance .csv files in {folder}")
    rows = [summarize(p) for p in tqdm(files, desc=f"Reading {Path(folder).name:12s}")]
    return pd.DataFrame(rows).sort_values("n_spacepoints")


def read_gpu_util(path):
    '''
    @brief reads a log of "nvidia-smi dmon -s u" (with or without -o T)
    @param path : Path of the log
    @return DataFrame with one row per second and the columns sm and mem
            (% of time the multiprocessors / the memory were busy). The idle
            seconds before the first and after the last busy one are removed:
            they are the start-up and the end of the program, not the event loop
    '''
    names, rows = None, []
    with open(path) as log:
        for line in log:
            fields = line.split()
            if not fields:
                continue
            if line.startswith("#"):
                # First header line: the column names (the second has the units)
                if names is None and "sm" in fields[1:]:
                    names = fields[1:] if fields[0] == "#" else [fields[0][1:]] + fields[1:]
                continue
            rows.append(fields)
    if names is None or not rows:
        return pd.DataFrame(columns=["sm", "mem"])
    # Rows shorter than the header are dropped: the last line can be cut when
    # nvidia-smi dmon is stopped while writing it
    rows = [r[:len(names)] for r in rows if len(r) >= len(names)]
    if not rows:
        return pd.DataFrame(columns=["sm", "mem"])
    data = pd.DataFrame(rows, columns=names)
    data = data[["sm", "mem"]].apply(pd.to_numeric, errors="coerce").dropna()
    busy = np.flatnonzero(data["sm"].to_numpy() > 0)
    if busy.size == 0:
        return data.iloc[0:0]
    return data.iloc[busy[0]:busy[-1] + 1]


def load_gpu_util(folder):
    '''
    @brief mean GPU utilization of every run of a folder of dmon logs
    @param folder : (str or Path) folder with one log per run (run_500.log, ...),
           or folder of a series in the layout of run_reco.sh
           (run_500/logs/gpu_util.log, ...)
    @return DataFrame with the columns particles, gpu_sm, gpu_mem, gpu_samples
    '''
    folder = Path(folder)
    paths = sorted(folder.glob("*.log")) + sorted(folder.glob("run_*/logs/gpu_util.log"))
    rows = []
    for path in tqdm(paths,
                     desc=f"GPU logs {Path(folder).name:11s}"):
        particles = run_number(path)
        if particles is None:
            continue
        data = read_gpu_util(path)
        rows.append({"particles": particles,
                     "gpu_sm": data["sm"].mean() if len(data) else np.nan,
                     "gpu_mem": data["mem"].mean() if len(data) else np.nan,
                     "gpu_samples": len(data)})
    if not rows:
        raise SystemExit(f"No GPU utilization logs in {folder}")
    return pd.DataFrame(rows)


def plot_gpu_util(series):
    '''
    @brief mean utilization of the GPU during the event loop against the
           event size, one line per series: multiprocessors (left) and memory
           (right)
    @param series : dict label -> summary DataFrame with the gpu_* columns
    @return the matplotlib figure
    '''
    fig, (ax_sm, ax_mem) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for label, s in series.items():
        ax_sm.plot(s["n_spacepoints"], s["gpu_sm"], marker="o", label=label)
        ax_mem.plot(s["n_spacepoints"], s["gpu_mem"], marker="o", label=label)
    for ax, what in ((ax_sm, "multiprocessors (sm)"), (ax_mem, "memory (mem)")):
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Spacepoints per event (median)")
        ax.set_ylabel(f"Mean GPU utilization, {what} [%]")
        ax.grid(True, which="both", alpha=0.3)
    ax_sm.legend()
    fig.suptitle("GPU utilization during the event loop (nvidia-smi dmon, 1 sample/s)")
    fig.tight_layout()
    return fig


def natural_key(path):
    '''
    @brief sorting key that compares the numbers in a name as numbers
           (threads_2 before threads_16)
    @param path : Path to sort
    @return list of strings and integers
    '''
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def has_runs(folder, backend="gpu"):
    '''
    @brief tells whether a folder contains performance files of runs
    @param folder : Path of the folder
    @param backend : gpu or cpu (traccc_reco), or hough (traccc_hough_reco)
    @return (bool) True if it holds at least one <backend>_run_<N>.csv or
            run_<N>/logs/<backend>_performance.csv
    '''
    return bool(performance_files(folder, backend))


def expand_folders(folders, backend="gpu"):
    '''
    @brief turns the folders given on the command line into the list of
           series: a folder with performance files is one series, and each
           of its subfolders with performance files (threads_2, threads_4,
           ...) is one more, in natural order
    @param folders : list of folder names
    @param backend : gpu or cpu (traccc_reco), or hough (traccc_hough_reco)
    @return list of Paths, without repetitions
    '''
    found = []
    for folder in map(Path, folders):
        if has_runs(folder, backend):
            found.append(folder)
        subfolders = sorted((d for d in folder.iterdir()
                             if d.is_dir() and has_runs(d, backend)),
                            key=natural_key) if folder.is_dir() else []
        found += subfolders
    unique = list(dict.fromkeys(p.resolve() for p in found))
    if not unique:
        raise SystemExit(f"No performance .csv files in {' '.join(folders)}")
    return unique


def series_label(folder):
    '''
    @brief legend label of a series: "N threads" for a threads_N folder,
           otherwise the folder name
    @param folder : Path of the series
    @return (str) the label
    '''
    match = re.fullmatch(r"threads_(\d+)", folder.name)
    if match:
        n = int(match.group(1))
        return f"{n} thread" + ("s" if n > 1 else "")
    return folder.name


def util_folder(folder):
    '''
    @brief folder of the nvidia-smi dmon logs that goes with a series: the
           series itself in the layout of run_reco.sh (run_N/logs/gpu_util.log),
           otherwise results/reco/gpu_util_tN for results/reco/threads_N
    @param folder : Path of the series
    @return Path of the logs, or None if there is none
    '''
    if any(folder.glob("run_*/logs/gpu_util.log")):
        return folder
    match = re.fullmatch(r"threads_(\d+)", folder.name)
    if match:
        candidate = folder.parent / f"gpu_util_t{match.group(1)}"
        if candidate.is_dir():
            return candidate
    return None


def plot_reco_time(series):
    '''
    @brief median reconstruction time per event against the event size, one
           line per series, with the 16-84% band of the events as error bars
    @param series : dict label -> summary DataFrame
    @return the matplotlib figure
    '''
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, s in series.items():
        err = [s["t_reco_ms"] - s["t_reco_lo_ms"], s["t_reco_hi_ms"] - s["t_reco_ms"]]
        ax.errorbar(s["n_spacepoints"], s["t_reco_ms"], yerr=err, marker="o",
                    capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spacepoints per event (median)")
    ax.set_ylabel("Reconstruction time per event [ms]")
    ax.set_title("Reconstruction time, file reading excluded")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_breakdown(label, s):
    '''
    @brief share of every step in the time of one event, for every run of a
           series: shows which step dominates as the events grow
    @param label : name of the series
    @param s : summary DataFrame of the series
    @return the matplotlib figure
    '''
    steps = [c for c in STEPS if c in s.columns]
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7, 5))
    bottom = np.zeros(len(s))
    # File reading first, hatched: it is part of the event time but not of
    # the reconstruction.
    ax.bar(x, s["t_read_ms"], color="lightgrey", hatch="//", edgecolor="grey",
           label="File reading")
    bottom += s["t_read_ms"].to_numpy()
    for step in steps:
        ax.bar(x, s[step], bottom=bottom, label=STEPS[step])
        bottom += s[step].to_numpy()
    ax.set_xticks(x, [f"{p}" if pd.notna(p) else r for p, r in zip(s["particles"], s["run"])])
    ax.set_xlabel("Particles per event")
    ax.set_ylabel("Median time per event [ms]")
    ax.set_title(f"Time per step ({label})")
    # Legend outside the axes, so that it never covers the bars
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


def plot_throughput(series):
    '''
    @brief events per second against the event size, one line per series
    @param series : dict label -> summary DataFrame (only those with throughput)
    @return the matplotlib figure
    '''
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, s in series.items():
        ax.plot(s["n_spacepoints"], s["events_per_s"], marker="o", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Spacepoints per event (median)")
    ax.set_ylabel("Throughput [events/s]")
    ax.set_title("Throughput, file reading included")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plots the performance files of the runs. Every folder is "
                    "one series (e.g. cpu, gpu, gpu with 4 threads).")
    parser.add_argument("folders", nargs="+",
                        help="folders with performance files (one <backend>_run_N.csv "
                             "per run, or one run_N/ folder per run as written by "
                             "run_reco.sh); the subfolders with performance files "
                             "(e.g. threads_N) are added too, so results/reco alone "
                             "gives every thread count")
    parser.add_argument("-b", "--backend", nargs="+", default=["gpu"],
                        choices=["gpu", "cpu", "hough"],
                        help="reconstructions to read: gpu or cpu (traccc_reco), hough "
                             "(traccc_hough_reco, run_N/hough/performance.csv); with "
                             "more than one, every folder gives one series per "
                             "reconstruction (default: gpu)")
    parser.add_argument("-l", "--labels", nargs="+",
                        help="legend label of every series (default: 'N threads' "
                             "for threads_N folders, otherwise the folder name)")
    parser.add_argument("-u", "--util", nargs="+",
                        help="folders with the nvidia-smi dmon logs, one per series and "
                             "in the same order (default: the run_N/logs/gpu_util.log "
                             "of the series, or gpu_util_tN next to threads_N)")
    parser.add_argument("-n", "--name", default=None,
                        help="name of the PDF (default: names of the series)")
    parser.add_argument("-o", "--output", default="results",
                        help="output folder of the PDF (default: results)")
    parser.add_argument("--show", action="store_true", help="also open the plots on screen")
    args = parser.parse_args()

    # One series per (folder, reconstruction)
    pairs = [(f, b) for b in args.backend for f in expand_folders(args.folders, b)]
    folders = [f for f, _ in pairs]
    backends = [b for _, b in pairs]
    if args.labels:
        labels = args.labels
    elif len(args.backend) > 1:
        labels = [f"{series_label(f)} ({b})" for f, b in pairs]
    else:
        labels = [series_label(f) for f in folders]
    if len(labels) != len(folders):
        raise SystemExit(f"Give one label per series: found {len(folders)} series ("
                         + ", ".join(f"{f} ({b})" for f, b in pairs) + ")")
    print("Series:")
    for label, (folder, backend) in zip(labels, pairs):
        print(f"  {label:20s} {folder} ({backend})")

    series = {label: load_series(folder, backend)
              for label, (folder, backend) in zip(labels, pairs)}

    # GPU utilization, matched to the runs by their number of particles (only
    # for the series reconstructed on the GPU)
    if args.util:
        if len(args.util) != len(folders):
            raise SystemExit("Give one folder of GPU logs per series")
        logs = [Path(u) for u in args.util]
    else:
        logs = [util_folder(f) if b == "gpu" else None for f, b in pairs]
    for label, folder in zip(labels, logs):
        if folder is not None:
            series[label] = series[label].merge(load_gpu_util(folder),
                                                on="particles", how="left")

    figs = [plot_reco_time(series)]
    figs += [plot_breakdown(label, s) for label, s in series.items()]
    with_throughput = {l: s for l, s in series.items()
                       if "events_per_s" in s and s["events_per_s"].notna().any()}
    if with_throughput:
        figs.append(plot_throughput(with_throughput))
    with_util = {l: s for l, s in series.items()
                 if "gpu_sm" in s and s["gpu_sm"].notna().any()}
    if with_util:
        figs.append(plot_gpu_util(with_util))

    # Default name: the folder given (results/reco -> reco.pdf), or the
    # labels of the series when several folders are given
    if args.name:
        name = args.name
    elif len(args.folders) == 1:
        name = Path(args.folders[0]).resolve().name
    else:
        name = "_vs_".join(l.replace(" ", "") for l in labels)
    output = Path(args.output) / f"{name}.pdf"
    save_plots(figs, output)

    # The numbers behind the plots, for the report.
    table = pd.concat(series, names=["series"]).reset_index(level=0)
    table.to_csv(output.with_suffix(".csv"), index=False, float_format="%.4g")
    print(f"Summary table saved in {output.with_suffix('.csv')}")

    if args.show:
        plt.show()