import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from tqdm import tqdm
except ImportError:  
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
    "t_hough_ms": "Hough transform",  
}


## Other measures of the event size, written by traccc_reco in its performance
## file, with their names for the plots: the time is also drawn against each
## of them (the Hough transform has none of them, only the spacepoints)
SIZE_VARIABLES = {
    "n_measurements": "Measurements",
    "n_seeds": "Seeds",
    "n_found_tracks": "Found tracks",
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
    @param path : Path of the performance file
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
    @param path : Path of the performance file
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
    @return (dict) particles, median spacepoints (and the other sizes of
            SIZE_VARIABLES, for traccc), median and 16-84% band of the
            reconstruction time, median time of every step, throughput
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
    # Measurements, seeds, found tracks: only in the files of traccc
    for column in SIZE_VARIABLES:
        if column in data.columns:
            row[column] = data[column].median()
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


def plot_reco_time(ax, series, x="n_spacepoints", xlabel="Spacepoints per event (median)"):
    '''
    @brief median reconstruction time per event against the event size, one
           line per series, with the 16-84% band of the events as error bars
    @param ax : matplotlib axes to draw on
    @param series : dict label -> summary DataFrame; the series without the
           column x are skipped, but keep their colour
    @param x : (str) column of the event size: n_spacepoints or one of
           SIZE_VARIABLES
    @param xlabel : (str) label of the x axis
    '''
    for i, (label, s) in enumerate(series.items()):
        if x not in s or s[x].isna().all():
            continue
        err = [s["t_reco_ms"] - s["t_reco_lo_ms"], s["t_reco_hi_ms"] - s["t_reco_ms"]]
        ax.errorbar(s[x], s["t_reco_ms"], yerr=err, marker="o", capsize=3,
                    color=f"C{i % 10}", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reconstruction time per event [ms]")
    ax.set_title("Time per event, file reading excluded")
    ax.grid(True, which="both", alpha=0.3)


def plot_time_vs(series, column):
    '''
    @brief time per event (left) and events per second (right) against one
           more measure of the event size (measurements, seeds or found
           tracks), for the series that have it (traccc): they show which
           quantity drives the time better than the spacepoints
    @param series : dict label -> summary DataFrame
    @param column : (str) one of SIZE_VARIABLES
    @return the matplotlib figure, or None if no series has the column
    '''
    with_column = {l: s for l, s in series.items()
                   if column in s and s[column].notna().any()}
    if not with_column:
        return None
    name = SIZE_VARIABLES[column]
    xlabel = f"{name} per event (median)"
    with_throughput = any("events_per_s" in s and s["events_per_s"].notna().any()
                          for s in with_column.values())
    if with_throughput:
        fig, (ax_time, ax_rate) = plt.subplots(1, 2, figsize=(12, 4.8))
        # All the series are passed, so that every one keeps its colour
        plot_throughput(ax_rate, series, x=column, xlabel=xlabel)
    else:
        fig, ax_time = plt.subplots(figsize=(7, 5))
    plot_reco_time(ax_time, series, x=column, xlabel=xlabel)
    fig.suptitle(f"Reconstruction time and throughput against the {name.lower()}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    handles, names = ax_time.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(names), 4), fontsize=9, frameon=False)
    return fig


def plot_breakdown(ax, label, s):
    '''
    @brief share of every step in the time of one event, for every run of a
           series, drawn on the given axes: shows which step dominates as the
           events grow
    @param ax : matplotlib axes to draw on
    @param label : name of the series, used as title of the axes
    @param s : summary DataFrame of the series
    '''
    steps = [c for c in STEPS if c in s.columns]
    x = np.arange(len(s))
    bottom = np.zeros(len(s))
    # File reading first, hatched: it is part of the event time but not of
    # the reconstruction.
    ax.bar(x, s["t_read_ms"], color="lightgrey", hatch="//", edgecolor="grey",
           label="File reading")
    bottom += s["t_read_ms"].to_numpy()
    for step in steps:
        # Same colour for a step in every panel, so that one legend is enough
        ax.bar(x, s[step], bottom=bottom, color=STEP_COLOURS[step], label=STEPS[step])
        bottom += s[step].to_numpy()
    ax.set_xticks(x, [f"{p}" if pd.notna(p) else r for p, r in zip(s["particles"], s["run"])],
                  rotation=45, fontsize=8)
    ax.set_xlabel("Particles per event")
    ax.set_title(label)


## Colour of every step, the same in all the panels of the breakdown
STEP_COLOURS = {step: f"C{i}" for i, step in enumerate(STEPS)}


def plot_breakdowns(series, ncols=4):
    '''
    @brief time per step of every series, one panel per series (ncols per
           row), with a single legend and a common title
    @param series : dict label -> summary DataFrame
    @param ncols : (int) panels per row
    @return the matplotlib figure
    '''
    n = len(series)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows),
                             squeeze=False)
    for ax, (label, s) in zip(axes.flat, series.items()):
        plot_breakdown(ax, label, s)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Median time per event [ms]")

    # One legend for all the panels, with every step found in at least one
    handles = {}
    for ax in axes.flat[:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.suptitle("Time per step", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), 6), fontsize=9,
               frameon=False)
    return fig


def plot_throughput(ax, series, x="n_spacepoints", xlabel="Spacepoints per event (median)"):
    '''
    @brief events per second against the event size, one line per series
    @param ax : matplotlib axes to draw on
    @param series : dict label -> summary DataFrame; the series without
           throughput or without the column x are skipped, but keep their colour
    @param x : (str) column of the event size: n_spacepoints or one of
           SIZE_VARIABLES
    @param xlabel : (str) label of the x axis
    '''
    for i, (label, s) in enumerate(series.items()):
        if "events_per_s" not in s or s["events_per_s"].isna().all():
            continue
        if x not in s or s[x].isna().all():
            continue
        ax.plot(s[x], s["events_per_s"], marker="o", color=f"C{i % 10}", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Throughput [events/s]")
    ax.set_title("Events per second, file reading included")
    ax.grid(True, which="both", alpha=0.3)


def plot_time_and_throughput(series):
    '''
    @brief time per event (left) and events per second (right) side by side,
           with one legend below them; without throughput only the time is drawn
    @param series : dict label -> summary DataFrame
    @return the matplotlib figure
    '''
    with_throughput = any("events_per_s" in s and s["events_per_s"].notna().any()
                          for s in series.values())
    if with_throughput:
        fig, (ax_time, ax_rate) = plt.subplots(1, 2, figsize=(12, 4.8))
        plot_throughput(ax_rate, series)
    else:
        fig, ax_time = plt.subplots(figsize=(7, 5))
    plot_reco_time(ax_time, series)
    fig.suptitle("Reconstruction time and throughput", fontsize=14, fontweight="bold")
    fig.tight_layout()
    handles, names = ax_time.get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(names), 4), fontsize=9, frameon=False)
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

    figs = [plot_time_and_throughput(series)]
    # The same against measurements, seeds and found tracks (traccc only)
    for column in SIZE_VARIABLES:
        fig = plot_time_vs(series, column)
        if fig is not None:
            figs.append(fig)
    figs.append(plot_breakdowns(series))

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