import pandas as pd
import argparse 
from pathlib import Path
import multiprocessing as mp
from uncertainties import ufloat
import matplotlib.pyplot as plt

val_names = ["efficiency", "fake_rate", "duplicate_rate"]

def extract_column(path, label_col_name, names):

    df = pd.read_csv(path)
    row = df.iloc[0]

    values = [ufloat(row[n], row[n + "_err"]) for n in names]

    return pd.DataFrame({row[label_col_name]: values}, index=val_names)

def plot_table(df, out_path):

    cell_text = df.map(lambda x: f"{x:.2uP}").values 

    fig, ax = plt.subplots(figsize=(2 + 2 * len(df.columns), 0.6 + 0.4 * len(df)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for (r, c), cell in table.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparison in reconstruction and timing performance between hough and traccc tracking algorithm")

    parser.add_argument("traccc_file", help="names of the analyzed files related to traccc")

    parser.add_argument("hough_file", help="names of the analyzed files related to hough")

    parser.add_argument("-o", "--output", type=Path , default=Path("results"), help="output folder (default : results)")

    parser.add_argument("-n", "--name", default="hough_traccc_comparison", help="name of the output file (default : hough_traccc_comparison)")

    args = parser.parse_args()

    with mp.Pool(2) as pool:
        col_hough, col_traccc = pool.starmap(extract_column,[(args.hough_file, "series", val_names),(args.traccc_file, "series", val_names)],)   

    result = pd.concat([col_hough, col_traccc], axis=1)

    args.output.mkdir(parents=True, exist_ok=True)
    plot_table(result, args.output / f"{args.name}.png")