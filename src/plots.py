import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

## Variables histogrammed by default
DEFAULT_VARIABLES = ["n_spacepoints", "t_wall_ms"]

## Axis labels of the variables; the others use the column name
LABELS = {
    "n_spacepoints": "# Spacepoints",
    "t_wall_ms": "Reconstruction time [ms]",
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
    parser.add_argument("-o", "--output", default="results",
                        help="output folder of the PDF (default: results)")
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

    # Output PDF named after the analysed folder (data/run_100ev/ -> run_100ev.pdf)
    folder_name = Path(args.foldername).resolve().name
    output = Path(args.output) / f"{folder_name}.pdf"

    # One DataFrame per run, without the warm-up event
    runs = [(path.stem, pd.read_csv(path).iloc[1:]) for path in files]

    figs = []

    # Overlaid histograms: one figure per variable
    for variable in args.variables:
        figs.append(plot_overlaid_histograms(runs, variable, args.bins, args.logx))

    # Mean reconstruction time of each run
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, data in runs:
        data_average = data.mean(numeric_only=True)
        ax.scatter(data_average["n_spacepoints"], data_average["t_wall_ms"], label=label, s=12)

    ax.set_xlabel(LABELS["n_spacepoints"])
    ax.set_ylabel(LABELS["t_wall_ms"])
    ax.legend(fontsize=7)

    fig.tight_layout()
    figs.append(fig)

    # All the plots in one PDF, one per page
    save_plots(figs, output)
    if args.show:
        plt.show()