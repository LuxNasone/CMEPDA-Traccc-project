import pandas as pd
import argparse 
from matplotlib import pyplot as plt
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Digitizes the measurements of the toy MC.")
    parser.add_argument("foldername", type=str, help="folder with performance files")
    args = parser.parse_args()

    path = args.foldername

    fig, (ax_seed, ax_wall) = plt.subplots(1, 2, figsize=(11, 4))

    for path in sorted(Path(args.foldername).glob("*.csv")):  
        data = pd.read_csv(path)
        data = data.iloc[1:]  
        data_average = data.mean(numeric_only=True)                             
        label = path.stem                                   
        ax_seed.scatter(data_average["n_spacepoints"], data_average["t_wall_ms"], label=label, s=12)
        ax_wall.scatter(data_average["n_cells"], data_average["t_wall_ms"], label=label, s=12)

    ax_seed.set_xlabel("# Spacepoints")
    ax_seed.set_ylabel("t[ms]")
    ax_wall.set_xlabel("# Cells")
    ax_wall.set_ylabel("t[ms]")

    for ax in (ax_seed, ax_wall):
        ax.legend(fontsize=7)

    fig.tight_layout()
    plt.show()
