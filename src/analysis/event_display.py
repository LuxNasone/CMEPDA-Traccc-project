'''
@brief Pictures of the simulated detector and events, for the documentation.

From the files written by the toy MC and by the digitizer it draws, in one PDF
(and optionally as PNG images for the README):
  - a first page with the parameters of the run: those of the configuration
    file of the simulation, if there is one, and those measured on the data
    (particles, pT, eta, vertex, layers, pixels, noise);
  - one event in the transverse (x-y) and longitudinal (r-z) views, with the
    true hits of every particle joined in order and coloured by its pT;
  - the hits of many events in the x-y and r-z views (where the detector is);
  - the phi-z hit map of some barrel layers;
  - the pixel occupancy of the modules of the same layers, from the cells.

The barrel layers are found from the hits themselves: the modules (geometry_id)
whose hits spread more along z than along r are barrel modules, and modules at
similar radius form a layer.

The configuration file is the *.conf found in the run folder, or the one given
with --config. Its values are shown as they are written in the file: options
given on the command line of the simulation (e.g. gen-nparticles=1000) are not
in it, so the values measured on the data are always shown as well.

Usage:
    python src/analysis/event_display.py <run folder> [--event N] [--max-events N]
                                         [--layers 0 2 -1] [--config FILE]
                                         [-o results] [--png-dir docs/images]
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

## Rows of the parameter table that fit in one page
ROWS_PER_PAGE = 36


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


def read_config_file(path):
    '''
    @brief reads a configuration file of "key = value" (or "--key=value") lines
    @param path : (Path) the file; "#" starts a comment
    @return (list) (key, value) pairs in the order of the file; a line
            without "=" is kept as a key with an empty value
    '''
    pairs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition("=")
        pairs.append((key.strip().lstrip("-"), value.strip()))
    return pairs


def find_config_file(folder):
    '''
    @brief configuration file of the simulation saved in a run folder
    @param folder : (Path) run folder
    @return (Path) the first *.conf of the folder, or None if there is none
    '''
    found = sorted(folder.glob("*.conf"))
    return found[0] if found else None


def describe_range(values, unit="", fmt=".3g"):
    '''
    @brief "min - max (median m)" text of a set of values
    @param values : array-like of numbers
    @param unit : (str) unit written after the numbers
    @param fmt : (str) format of the numbers
    @return (str) the text, or "-" if there are no values
    '''
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "-"
    unit = f" {unit}" if unit else ""
    return (f"{values.min():{fmt}} - {values.max():{fmt}}{unit} "
            f"(median {np.median(values):{fmt}}{unit})")


def run_parameters(folder, events, map_events, event, particles, hits, layers, radii,
                   cells=None, config=None):
    '''
    @brief parameters of a run, for the first page of the PDF
    @param folder : (Path) run folder
    @param events : (list of int) all the events of the run
    @param map_events : (list of int) events used for the maps
    @param event : (int) event drawn alone
    @param particles : DataFrame of the particles_initial of map_events, with
           an "event" column
    @param hits : DataFrame of the hits of map_events, with r
    @param layers : (dict) barrel layers, as from find_barrel_layers
    @param radii : (list) radii of the barrel layers
    @param cells : DataFrame of the cells of map_events, or None if not read
    @param config : (Path) configuration file of the simulation, or None
    @return (dict) section title -> list of (name, value) pairs
    '''
    sections = {}
    sections["Run"] = [
        ("Folder", str(folder.resolve())),
        ("Events in the folder", f"{len(events)} ({events[0]} - {events[-1]})"),
        ("Events used for the maps", str(len(map_events))),
        ("Event drawn alone", str(event)),
    ]

    per_event = particles.groupby("event").size()
    pt = np.hypot(particles["px"], particles["py"])
    eta = np.arcsinh(particles["pz"] / pt.where(pt > 0))
    phi = np.arctan2(particles["py"], particles["px"])
    measured = [
        ("Particles per event", describe_range(per_event, fmt=".0f")),
        ("pT", describe_range(pt, "GeV")),
        ("eta", describe_range(eta)),
        ("phi", describe_range(phi, "rad")),
    ]
    if "q" in particles.columns:
        positive = float((particles["q"] > 0).mean())
        measured.append(("Charge", f"{positive:.1%} positive, {1 - positive:.1%} negative"))
    for axis in ("vx", "vy", "vz"):
        if axis in particles.columns:
            measured.append((f"Vertex {axis[1]} (mean, std)",
                             f"{particles[axis].mean():.3g}, {particles[axis].std():.3g} mm"))
    hits_per_particle = hits.groupby(["event", "particle_id"]).size()
    measured.append(("Hits per particle", describe_range(hits_per_particle, fmt=".0f")))
    measured.append(("Hits per event", describe_range(hits.groupby("event").size(), fmt=".0f")))
    sections["Simulation, measured on the data"] = measured

    detector = [
        ("Modules with hits", str(hits["geometry_id"].nunique())),
        ("Barrel layers", str(len(layers))),
    ]
    if radii:
        detector.append(("Barrel radii", ", ".join(f"{r:.0f}" for r in radii) + " mm"))
    if cells is not None and not cells.empty:
        noise = cells["measurement_id"] == NOISE_ID
        detector += [
            ("Pixel matrix (channel0 x channel1)",
             f"{int(cells['channel0'].max()) + 1} x {int(cells['channel1'].max()) + 1}"),
            ("Cells per event", describe_range(cells.groupby("event").size(), fmt=".0f")),
            ("Noise cells", f"{noise.mean():.2%} of the cells"),
        ]
    sections["Detector and digitization"] = detector

    if config is not None:
        sections[f"Configuration file: {config}"] = (
            read_config_file(config) or [("(empty file)", "")])
    else:
        sections["Configuration file"] = [
            ("not found", "no *.conf in the run folder; give one with --config")]
    return sections


def draw_parameter_pages(title, sections, rows_per_page=ROWS_PER_PAGE):
    '''
    @brief the parameters as a two-column table, split in as many pages as needed
    @param title : (str) title of the pages
    @param sections : (dict) section title -> list of (name, value) pairs
    @param rows_per_page : (int) rows of one page, section titles included
    @return list of matplotlib figures
    '''
    rows = []
    for section, pairs in sections.items():
        rows.append((section, None))
        rows += [(str(name), str(value)) for name, value in pairs]

    pages = [rows[i:i + rows_per_page] for i in range(0, len(rows), rows_per_page)]
    figs = []
    for number, page in enumerate(pages, start=1):
        fig = plt.figure(figsize=(8.27, 11.69))   # A4
        suffix = f" ({number}/{len(pages)})" if len(pages) > 1 else ""
        fig.text(0.06, 0.96, title + suffix, fontsize=14, weight="bold", va="top")
        y, step = 0.92, 0.88 / rows_per_page
        for name, value in page:
            if value is None:   # section title
                fig.text(0.06, y, name, fontsize=11, weight="bold", va="top")
                fig.add_artist(plt.Line2D([0.06, 0.94], [y - 0.017, y - 0.017],
                                          color="grey", lw=0.5))
            else:
                fig.text(0.08, y, name, fontsize=9, va="top")
                fig.text(0.42, y, value, fontsize=9, va="top", family="monospace",
                         wrap=True)
            y -= step
        figs.append(fig)
    return figs


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
    parser.add_argument("--config", type=Path, default=None,
                        help="configuration file of the simulation to show in the "
                             "parameter page (default: the *.conf of the run folder)")
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
    particles = read_many(args.folder, map_events, "particles_initial")

    config = args.config if args.config is not None else find_config_file(args.folder)
    if config is not None and not config.is_file():
        parser.error(f"configuration file not found: {config}")

    figs = [("event_xy", draw_event(one, pt, radii, "xy")),
            ("event_rz", draw_event(one, pt, radii, "rz")),
            ("hits_xy", draw_hit_density(hits, "xy", len(map_events))),
            ("hits_rz", draw_hit_density(hits, "rz", len(map_events)))]

    cells = None
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

    # Parameter page(s) first, so that every PDF says how it was made
    sections = run_parameters(args.folder, events, map_events, event, particles, hits,
                              layers, radii, cells, config)
    pages = draw_parameter_pages(f"Run {args.folder.resolve().name}: parameters", sections)
    figs = [(f"parameters_{i}", fig) for i, fig in enumerate(pages, start=1)] + figs

    save(figs, args.output / f"{args.folder.resolve().name}_display.pdf", args.png_dir)


if __name__ == "__main__":
    main()