import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

## Variables histogrammed by default
DEFAULT_VARIABLES = ["n_spacepoints", "t_wall_ms"]

## Axis labels of the variables; the others use the column name
LABELS = {
    "n_spacepoints": "# Spacepoints",
    "t_wall_ms": "Reconstruction time [ms]",
}


def save_plot(fig, output):
    '''
    @brief saves the figure, overwriting the file if it already exists
    @param fig : matplotlib figure to save
    @param output : (str or Path) output file; the extension sets the format (.png, .pdf, ...)
    '''
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)   # create the folder if missing
    fig.savefig(output, dpi=150, bbox_inches="tight")  # savefig overwrites by default
    print(f"Plot saved in {output}")


def run_number(path):
    '''
    @brief number of particles of a run, read from its name (gpu_run_500.csv -> 500)
    @param path : Path of the performance file
    @return (int) the number in the name, or 0 if there is none
    '''
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def plot_overlaid_histograms(runs, variable, bins=60, log_x=False):
    '''
    @brief draws the histogram of one variable for every run, overlaid in one figure
    @param runs : list of (label, DataFrame) pairs, one per run
    @param variable : column to histogram, e.g. "t_wall_ms"
    @param bins : number of bins, the same for all the runs
    @param log_x : if True, logarithmic x axis and logarithmic bins
    @return the matplotlib figure
    '''
    values = pd.concat([data[variable] for _, data in runs])
    low, high = values.min(), values.max()
    # The same bins for all the runs, so that the histograms can be compared
    if log_x:
        edges = np.geomspace(max(low, 1e-6), high, bins + 1)
    else:
        edges = np.linspace(low, high, bins + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, data in runs:
        ax.hist(data[variable], bins=edges, histtype="step", linewidth=1.5, label=label)

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(LABELS.get(variable, variable))
    ax.set_ylabel("events")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plots the performance files of the runs.")
    parser.add_argument("foldername", type=str, help="folder with performance files")
    parser.add_argument("-o", "--output", default="results/plots",
                        help="output folder of the plots (default: results/plots)")
    parser.add_argument("-v", "--variables", nargs="+", default=DEFAULT_VARIABLES,
                        help="variables to histogram (default: n_spacepoints t_wall_ms)")
    parser.add_argument("--bins", type=int, default=60, help="bins of the histograms (default: 60)")
    parser.add_argument("--logx", action="store_true", help="logarithmic x axis")
    parser.add_argument("--show", action="store_true", help="also open the plots on screen")
    args = parser.parse_args()

    # Runs sorted by number of particles (run_50 before run_100)
    files = sorted(Path(args.foldername).glob("*.csv"), key=run_number)

    if not files:
        raise SystemExit(f"No .csv files in {args.foldername}")

    output = Path(args.output)

    # One DataFrame per run, without the warm-up event
    runs = [(path.stem, pd.read_csv(path).iloc[1:]) for path in files]

    # Overlaid histograms: one figure per variable
    for variable in args.variables:
        hist_fig = plot_overlaid_histograms(runs, variable, args.bins, args.logx)
        save_plot(hist_fig, output / f"hist_{variable}.png")

    # Mean reconstruction time of each run
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, data in runs:
        data_average = data.mean(numeric_only=True)
        ax.scatter(data_average["n_spacepoints"], data_average["t_wall_ms"], label=label, s=12)

    ax.set_xlabel(LABELS["n_spacepoints"])
    ax.set_ylabel(LABELS["t_wall_ms"])
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_plot(fig, output / "reconstruction_time.png")
    if args.show:
        plt.show()
