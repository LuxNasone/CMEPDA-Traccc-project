'''
@brief Pictures of the simulated detector and events, for the documentation.

From the files written by the toy MC and by the digitizer it draws, in one PDF
(and optionally as PNG images for the README):
  - one event in the transverse (x-y) and longitudinal (r-z) views, with the
    true hits of every particle joined in order and coloured by its pT;
  - the hits of many events in the x-y and r-z views (where the detector is);
  - the phi-z hit map of some barrel layers;
  - the pixel occupancy of the modules of the same layers, from the cells.

The barrel layers are found from the hits themselves: the modules (geometry_id)
whose hits spread more along z than along r are barrel modules, and modules at
similar radius form a layer.

Usage:
    python src/analysis/event_display.py <run folder> [--event N] [--max-events N]
                                         [--layers 0 2 -1] [-o results] [--png-dir docs/images]
'''

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors, pyplot as plt, ticker
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

## measurement_id of the cells that come from noise (2^64 - 1)
NOISE_ID = 18446744073709551615

## Minimum radial gap (mm) between two barrel layers
LAYER_GAP_MM = 10.0


def event_file(folder, event, kind):
    '''
    @brief path of one file of an event, e.g. event000000003-hits.csv
    @param folder : (Path) run folder
    @param event : (int) event number
    @param kind : (str) hits, cells, particles_initial, ...
    @return (Path) the file
    '''
    return folder / f"event{event:09d}-{kind}.csv"


def available_events(folder):
    '''
    @brief numbers of the events of a run that have a hits file
    @param folder : (Path) run folder
    @return (list of int) sorted event numbers
    '''
    return sorted(int(p.name[5:14]) for p in folder.glob("event*-hits.csv"))


def read_many(folder, events, kind, columns=None):
    '''
    @brief reads one kind of file for several events into one table
    @param folder : (Path) run folder
    @param events : (list of int) event numbers
    @param kind : (str) hits, cells, ...
    @param columns : (list of str) columns to keep, None = all
    @return DataFrame with an extra "event" column
    '''
    tables = []
    for event in tqdm(events, desc=f"Reading {kind:17s}", leave=True):
        path = event_file(folder, event, kind)
        if path.exists():
            table = pd.read_csv(path, usecols=columns)
            table["event"] = event
            tables.append(table)
    if not tables:
        raise SystemExit(f"No {kind} files in {folder}")
    return pd.concat(tables, ignore_index=True)


def add_cylindrical(hits):
    '''
    @brief adds r (transverse radius) and phi to a table with tx, ty, tz
    @param hits : DataFrame of hits
    @return the same DataFrame, with the columns r and phi
    '''
    hits["r"] = np.hypot(hits["tx"], hits["ty"])
    hits["phi"] = np.arctan2(hits["ty"], hits["tx"])
    return hits


def find_barrel_layers(hits, min_hits=3):
    '''
    @brief groups the barrel modules into layers
    @param hits : DataFrame of hits of many events, with r
    @param min_hits : (int) modules with fewer hits are not classified
    @return (dict) layer index (0 = innermost) -> set of geometry_id, and
            (list) median radius of every layer
    '''
    modules = hits.groupby("geometry_id").agg(
        r=("r", "median"), r_range=("r", np.ptp), z_range=("tz", np.ptp),
        n=("r", "size"))
    modules = modules[modules["n"] >= min_hits]
    # A barrel module is a surface of (almost) fixed radius, long along z
    barrel = modules[modules["z_range"] > modules["r_range"]].sort_values("r")
    if barrel.empty:
        return {}, []

    # New layer wherever the radius jumps by more than LAYER_GAP_MM
    layer_of = np.concatenate([[0], np.cumsum(np.diff(barrel["r"]) > LAYER_GAP_MM)])
    layers, radii = {}, []
    for index in np.unique(layer_of):
        members = barrel[layer_of == index]
        layers[int(index)] = set(members.index)
        radii.append(float(members["r"].median()))
    return layers, radii


def particle_pt(particles):
    '''
    @brief transverse momentum of every particle
    @param particles : DataFrame of particles_initial
    @return (Series) pT in GeV, indexed by particle_id
    '''
    return pd.Series(np.hypot(particles["px"], particles["py"]).to_numpy(),
                     index=particles["particle_id"])


def draw_event(hits, pt, radii, view):
    '''
    @brief one event: the hits of every particle joined in order of distance
           from the origin, coloured by the pT of the particle
    @param hits : DataFrame of the hits of one event, with r
    @param pt : (Series) pT of the particles, by particle_id
    @param radii : (list) radii of the barrel layers, drawn as references
    @param view : "xy" (transverse) or "rz" (longitudinal)
    @return the matplotlib figure
    '''
    fig, ax = plt.subplots(figsize=(6.5, 6) if view == "xy" else (8, 5))
    norm = colors.LogNorm(vmin=max(pt.min(), 1e-3), vmax=max(pt.max(), 1e-3 * 1.01))
    cmap = plt.get_cmap("viridis")

    if view == "xy":
        for radius in radii:
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="lightgrey", lw=0.8))
    else:
        for radius in radii:
            ax.axhline(radius, color="lightgrey", lw=0.8)

    distance = np.sqrt(hits["tx"] ** 2 + hits["ty"] ** 2 + hits["tz"] ** 2)
    for pid, track in hits.assign(d=distance).sort_values("d").groupby("particle_id"):
        colour = cmap(norm(pt.get(pid, pt.min())))
        x, y = (track["tx"], track["ty"]) if view == "xy" else (track["tz"], track["r"])
        # The track starts at the vertex (the origin of the toy MC)
        ax.plot(np.r_[0, x], np.r_[0, y], color=colour, lw=1, alpha=0.8)
        ax.scatter(x, y, color=colour, s=8, zorder=3)

    if view == "xy":
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    else:
        ax.set_xlabel("z [mm]")
        ax.set_ylabel("r [mm]")
    event = int(hits["event"].iloc[0])
    ax.set_title(f"Event {event}: {hits['particle_id'].nunique()} particles, "
                 f"{len(hits)} hits ({'transverse' if view == 'xy' else 'longitudinal'} view)")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="pT [GeV]")
    # Plain numbers (0.5, 2, 6) instead of powers of ten on the log scale
    bar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    bar.ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%g"))
    fig.tight_layout()
    return fig


def draw_hit_density(hits, view, n_events):
    '''
    @brief hits of many events as a 2D histogram: shows where the detector is
    @param hits : DataFrame of hits, with r
    @param view : "xy" or "rz"
    @param n_events : (int) number of events, for the title
    @return the matplotlib figure
    '''
    fig, ax = plt.subplots(figsize=(6.5, 6) if view == "xy" else (8, 5))
    x, y = (hits["tx"], hits["ty"]) if view == "xy" else (hits["tz"], hits["r"])
    _, _, _, image = ax.hist2d(x, y, bins=400, norm=colors.LogNorm(), cmap="magma")
    if view == "xy":
        ax.set_aspect("equal")
    ax.set_xlabel("x [mm]" if view == "xy" else "z [mm]")
    ax.set_ylabel("y [mm]" if view == "xy" else "r [mm]")
    ax.set_title(f"Hits of {n_events} events ({'transverse' if view == 'xy' else 'longitudinal'} view)")
    fig.colorbar(image, ax=ax, label="hits per bin")
    fig.tight_layout()
    return fig


def draw_layer_map(hits, modules, radius, index, n_events):
    '''
    @brief phi-z map of the hits of one barrel layer
    @param hits : DataFrame of hits, with r and phi
    @param modules : (set) geometry_id of the modules of the layer
    @param radius : (float) radius of the layer, for the title
    @param index : (int) layer index (0 = innermost)
    @param n_events : (int) number of events, for the title
    @return the matplotlib figure
    '''
    layer = hits[hits["geometry_id"].isin(modules)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _, _, _, image = ax.hist2d(layer["tz"], layer["phi"], bins=(200, 180),
                               cmap="viridis", cmin=1)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("phi [rad]")
    ax.set_title(f"Barrel layer {index} (r = {radius:.0f} mm): {len(modules)} modules, "
                 f"{len(layer)} hits in {n_events} events")
    fig.colorbar(image, ax=ax, label="hits per bin")
    fig.tight_layout()
    return fig


def draw_occupancy(cells, modules, index, n_events):
    '''
    @brief pixel occupancy of the modules of one layer: the cells of all its
           modules and events on one pixel matrix, signal and noise apart
    @param cells : DataFrame of cells
    @param modules : (set) geometry_id of the modules of the layer
    @param index : (int) layer index, for the title
    @param n_events : (int) number of events, for the title
    @return the matplotlib figure
    '''
    layer = cells[cells["geometry_id"].isin(modules)]
    noise = layer["measurement_id"] == NOISE_ID
    fig, ax = plt.subplots(figsize=(5, 8))
    n0, n1 = int(cells["channel0"].max()) + 1, int(cells["channel1"].max()) + 1
    _, _, _, image = ax.hist2d(layer["channel0"], layer["channel1"],
                               bins=(min(n0, 120), min(n1, 400)),
                               range=((0, n0), (0, n1)), cmap="viridis", cmin=1)
    ax.set_xlabel("pixel column (channel0)")
    ax.set_ylabel("pixel row (channel1)")
    ax.set_title(f"Layer {index}: occupancy of {len(modules)} modules, {n_events} events\n"
                 f"{(~noise).sum()} signal cells, {noise.sum()} noise cells", fontsize=10)
    fig.colorbar(image, ax=ax, label="cells per bin")
    fig.tight_layout()
    return fig


def save(figs, pdf_path, png_dir=None):
    '''
    @brief saves the figures in one PDF, one per page, and optionally as PNG
    @param figs : list of (name, figure) pairs
    @param pdf_path : (Path) PDF file
    @param png_dir : (Path) folder of the PNG images, None = no PNG
    '''
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for _, fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    print(f"{len(figs)} plots saved in {pdf_path}")
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            fig.savefig(png_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"PNG images saved in {png_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pictures of the detector and of the events.")
    parser.add_argument("folder", type=Path, help="run folder written by the simulation")
    parser.add_argument("--event", type=int, default=None,
                        help="event to draw (default: the first one)")
    parser.add_argument("--max-events", type=int, default=100,
                        help="events used for the hit maps (default: 100)")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, -1],
                        help="barrel layers of the maps; negative = from the outside "
                             "(default: 0 -1, innermost and outermost)")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"),
                        help="output folder of the PDF (default: results)")
    parser.add_argument("--png-dir", type=Path, default=None,
                        help="also save every plot as PNG here, e.g. docs/images")
    args = parser.parse_args()

    events = available_events(args.folder)
    if not events:
        parser.error(f"no event*-hits.csv files in {args.folder}")
    event = events[0] if args.event is None else args.event
    if event not in events:
        parser.error(f"event {event} not found in {args.folder}")
    map_events = events[:args.max_events]

    hits = add_cylindrical(read_many(args.folder, map_events, "hits",
                                     ["particle_id", "geometry_id", "tx", "ty", "tz"]))
    layers, radii = find_barrel_layers(hits)
    print(f"Barrel layers found: {len(layers)}, radii (mm): "
          + ", ".join(f"{r:.0f}" for r in radii))

    one = add_cylindrical(read_many(args.folder, [event], "hits",
                                    ["particle_id", "geometry_id", "tx", "ty", "tz"]))
    pt = particle_pt(pd.read_csv(event_file(args.folder, event, "particles_initial")))

    figs = [("event_xy", draw_event(one, pt, radii, "xy")),
            ("event_rz", draw_event(one, pt, radii, "rz")),
            ("hits_xy", draw_hit_density(hits, "xy", len(map_events))),
            ("hits_rz", draw_hit_density(hits, "rz", len(map_events)))]

    if layers:
        chosen = sorted({i % len(layers) for i in args.layers if -len(layers) <= i < len(layers)})
        cells = read_many(args.folder, map_events, "cells",
                          ["geometry_id", "measurement_id", "channel0", "channel1"])
        for i in chosen:
            figs.append((f"layer{i}_phi_z",
                         draw_layer_map(hits, layers[i], radii[i], i, len(map_events))))
            figs.append((f"layer{i}_occupancy",
                         draw_occupancy(cells, layers[i], i, len(map_events))))
    else:
        print("No barrel layer found: the layer maps are skipped")

    save(figs, args.output / f"{args.folder.resolve().name}_display.pdf", args.png_dir)


if __name__ == "__main__":
    main()