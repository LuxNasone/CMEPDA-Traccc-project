import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
}


def save_plots(figs, output):
    '''
    @brief saves the figures in a single PDF file, one per page, overwriting it if it exists
    @param figs : list of matplotlib figures to save, in page order
    @param output : (str or Path) output PDF file
    '''
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)   # create the folder if missing
    with PdfPages(output) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    print(f"{len(figs)} plots saved in {output}")


def run_number(path):
    '''
    @brief number of particles of a run, read from its name (gpu_run_500.csv -> 500)
    @param path : Path of the performance file
    @return (int) the number in the name, or None if there is none
    '''
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


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
    return data.iloc[1:]   # old files, one thread: the first row


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
    # Reconstruction time: from the cells in memory to the fitted tracks in
    # memory, i.e. without reading the file and writing the CSV.
    t_reco = data[steps].sum(axis=1)

    row = {
        "run": path.stem,
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

    # Throughput: events finished per second of loop time (needs the
    # per-event start/end times of the multi-thread version).
    if {"t_start_ms", "t_end_ms"}.issubset(data.columns):
        span = data["t_end_ms"].max() - data["t_start_ms"].min()
        row["events_per_s"] = 1000. * len(data) / span if span > 0 else np.nan
    return row


def load_series(folder):
    '''
    @brief summarizes every performance file of a folder, one row per run
    @param folder : (str or Path) folder with the .csv performance files
    @return DataFrame sorted by number of spacepoints
    '''
    files = [p for p in Path(folder).glob("*.csv") if run_number(p) is not None]
    if not files:
        raise SystemExit(f"No performance .csv files in {folder}")
    return pd.DataFrame([summarize(p) for p in files]).sort_values("n_spacepoints")


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
    data = pd.DataFrame([r[:len(names)] for r in rows], columns=names)
    data = data[["sm", "mem"]].apply(pd.to_numeric, errors="coerce").dropna()
    busy = np.flatnonzero(data["sm"].to_numpy() > 0)
    if busy.size == 0:
        return data.iloc[0:0]
    return data.iloc[busy[0]:busy[-1] + 1]


def load_gpu_util(folder):
    '''
    @brief mean GPU utilization of every run of a folder of dmon logs
    @param folder : (str or Path) folder with one log per run (run_500.log, ...)
    @return DataFrame with the columns particles, gpu_sm, gpu_mem, gpu_samples
    '''
    rows = []
    for path in Path(folder).glob("*.log"):
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
    parser.add_argument("folders", nargs="+", help="folders with performance files")
    parser.add_argument("-l", "--labels", nargs="+",
                        help="legend label of every folder (default: folder names)")
    parser.add_argument("-u", "--util", nargs="+",
                        help="folders with the nvidia-smi dmon logs, one per folder of "
                             "performance files and in the same order (optional)")
    parser.add_argument("-n", "--name", default=None,
                        help="name of the PDF (default: names of the folders)")
    parser.add_argument("-o", "--output", default="results",
                        help="output folder of the PDF (default: results)")
    parser.add_argument("--show", action="store_true", help="also open the plots on screen")
    args = parser.parse_args()

    labels = args.labels or [Path(f).resolve().name for f in args.folders]
    if len(labels) != len(args.folders):
        raise SystemExit("Give one label per folder")

    series = {label: load_series(folder) for label, folder in zip(labels, args.folders)}

    # GPU utilization, matched to the runs by their number of particles
    if args.util:
        if len(args.util) != len(args.folders):
            raise SystemExit("Give one folder of GPU logs per folder of performance files")
        for label, folder in zip(labels, args.util):
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

    name = args.name or "_vs_".join(labels)
    output = Path(args.output) / f"{name}.pdf"
    save_plots(figs, output)

    # The numbers behind the plots, for the report.
    table = pd.concat(series, names=["series"]).reset_index(level=0)
    table.to_csv(output.with_suffix(".csv"), index=False, float_format="%.4g")
    print(f"Summary table saved in {output.with_suffix('.csv')}")

    if args.show:
        plt.show()