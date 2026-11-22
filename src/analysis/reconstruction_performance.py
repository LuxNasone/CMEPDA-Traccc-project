'''
@brief Physics performance of the reconstruction: efficiency, fake and
       duplicate rates, resolution of the track parameters, fit quality.

The reconstructed tracks are compared with the truth of the simulation:
  1. every reconstructed measurement (module + local position) is matched to the
     closest true measurement on the same module, within --tolerance-mm;
  2. a true measurement leads to its simulated hit (measurement-simhit-map) and
     the hit to its particle;
  3. a track belongs to the particle that produced at least --min-fraction of its
     measurements; a track with no such particle is a fake, a second track of the
     same particle is a duplicate;
  4. a particle is reconstructable if it is charged, has pT above --min-pt and at
     least --min-hits true measurements; the efficiency is the fraction of them
     with at least one track.
The track parameters (phi, theta, q/p, hence pT) are those of the fit on the
surface of the first measurement of the track: phi and theta are the direction
of the momentum where the track crosses that surface (the innermost hit), not
at the vertex. They are compared with the true momentum of the simulated hit on
the same surface.

If the tracks file has the covariance of the parameters (columns cov_<a>_<b>,
written by csv_writer.hpp), the errors of the fit are checked with the pulls of
phi, theta, q/p and pT, (fit - true) / error, which must be Gaussian with mean
0 and width 1 if the errors are right.

Input, one of:
  - a single run: its tracks folder (e.g. data/reco_gpu, right after the
    reconstruction) and its simulated run folder with the truth;
  - several runs: the folders <backend>_run_<N>_tracks of a results folder, and
    the folder with the simulated runs (run_<N>).

Usage:
    python src/analysis/physics_performance.py --tracks data/reco_gpu --truth data/run_100ev/run_500
    python src/analysis/physics_performance.py <results folder>... --data data/run_100ev
'''

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

## Name of the track folders written by run_reco.sh: <backend>_run_<N>_tracks
TRACKS_FOLDER = re.compile(r"(?P<backend>\w+?)_(?P<run>run_(?P<n>\d+))_tracks")

## Name of the event files: event000000003-<kind>.csv
EVENT_FILE = re.compile(r"event(\d+)-tracks\.csv")


def event_path(folder, event, kind):
    '''
    @brief path of one file of an event
    @param folder : (Path) folder of the event files
    @param event : (int) event number
    @param kind : (str) hits, measurements, tracks, ...
    @return (Path) the file
    '''
    return folder / f"event{event:09d}-{kind}.csv"


def read_truth(folder, event):
    '''
    @brief truth of one event: true measurements with their particle, and particles
    @param folder : (Path) simulated run folder
    @param event : (int) event number
    @return (measurements, particles): measurements with geometry_id, local0,
            local1, particle_id and the true momentum (tpx, tpy, tpz) of the hit;
            particles with particle_id, q, pt, eta, p
    '''
    meas = pd.read_csv(event_path(folder, event, "measurements"),
                       usecols=["measurement_id", "geometry_id", "local0", "local1"])
    links = pd.read_csv(event_path(folder, event, "measurement-simhit-map"))
    hits = pd.read_csv(event_path(folder, event, "hits"),
                       usecols=["particle_id", "tpx", "tpy", "tpz"])
    hits["hit_id"] = np.arange(len(hits))   # hit_id = row of the hits file
    meas = meas.merge(links, on="measurement_id").merge(hits, on="hit_id")

    particles = pd.read_csv(event_path(folder, event, "particles_initial"),
                            usecols=["particle_id", "px", "py", "pz", "q"])
    particles["pt"] = np.hypot(particles["px"], particles["py"])
    particles["p"] = np.sqrt(particles["pt"] ** 2 + particles["pz"] ** 2)
    particles["eta"] = np.arcsinh(particles["pz"] / particles["pt"].where(particles["pt"] > 0))
    n_meas = meas.groupby("particle_id").size().rename("n_true_meas")
    particles = particles.merge(n_meas, left_on="particle_id", right_index=True, how="left")
    particles["n_true_meas"] = particles["n_true_meas"].fillna(0).astype(int)
    return meas, particles


def match_measurements(reco, truth, tolerance):
    '''
    @brief matches every reconstructed measurement to the closest true measurement
           on the same module
    @param reco : DataFrame of the track measurements (track_id, geometry_id, local0, local1)
    @param truth : DataFrame of the true measurements (read_truth)
    @param tolerance : (float) maximum distance in mm
    @return reco with the columns particle_id (NaN if not matched), dist and the
            true momentum of the matched hit
    '''
    reco = reco.reset_index(drop=True).rename_axis("reco_row").reset_index()
    pairs = reco.merge(truth, on="geometry_id", suffixes=("", "_true"))
    pairs["dist"] = np.hypot(pairs["local0"] - pairs["local0_true"],
                             pairs["local1"] - pairs["local1_true"])
    pairs = pairs[pairs["dist"] <= tolerance]
    best = pairs.loc[pairs.groupby("reco_row")["dist"].idxmin(),
                     ["reco_row", "particle_id", "dist", "tpx", "tpy", "tpz"]]
    return reco.merge(best, on="reco_row", how="left")


def analyse_event(tracks_folder, truth_folder, event, args):
    '''
    @brief matches the tracks of one event to the particles
    @param tracks_folder : (Path) folder with the reconstructed tracks
    @param truth_folder : (Path) simulated run folder
    @param event : (int) event number
    @param args : command line options (tolerance_mm, min_fraction, min_pt, min_hits)
    @return (tracks, particles): one row per track with its particle, purity,
            category (matched / duplicate / fake) and true parameters; one row per
            particle with reconstructable and found
    '''
    truth, particles = read_truth(truth_folder, event)
    tracks = pd.read_csv(event_path(tracks_folder, event, "tracks"))
    reco = pd.read_csv(event_path(tracks_folder, event, "track-measurements"))
    reco = match_measurements(reco, truth, args.tolerance_mm)

    # Majority particle of every track and the fraction of its measurements
    counts = (reco.dropna(subset=["particle_id"])
                  .groupby(["track_id", "particle_id"]).size().rename("n").reset_index())
    best = counts.loc[counts.groupby("track_id")["n"].idxmax()]
    n_meas = reco.groupby("track_id").agg(n_meas=("geometry_id", "size"),
                                          n_meas_matched=("particle_id", "count"))
    tracks = (tracks.merge(n_meas, left_on="track_id", right_index=True, how="left")
                    .merge(best[["track_id", "particle_id", "n"]], on="track_id", how="left"))
    tracks["purity"] = (tracks["n"] / tracks["n_meas"]).fillna(0.0)
    tracks["matched"] = tracks["purity"] >= args.min_fraction
    tracks.loc[~tracks["matched"], "particle_id"] = np.nan

    # First matched track of a particle = matched, the others = duplicates
    tracks["category"] = np.where(tracks["matched"], "matched", "fake")
    dup = tracks["matched"] & tracks.duplicated("particle_id", keep="first")
    tracks.loc[dup, "category"] = "duplicate"

    # Truth on the surface of the parameters: the first measurement of the track
    first = reco.groupby("track_id").first()[["tpx", "tpy", "tpz"]]
    tracks = tracks.merge(first, left_on="track_id", right_index=True, how="left")
    tracks = tracks.merge(particles[["particle_id", "q", "pt", "eta"]],
                          on="particle_id", how="left", suffixes=("", "_true"))

    particles["reconstructable"] = ((particles["q"] != 0)
                                    & (particles["pt"] > args.min_pt)
                                    & (particles["n_true_meas"] >= args.min_hits))
    found = set(tracks.loc[tracks["matched"], "particle_id"].dropna().astype(int))
    particles["found"] = particles["particle_id"].isin(found)
    tracks["event"] = event
    particles["event"] = event
    return tracks, particles


def add_residuals(tracks):
    '''
    @brief residuals of the fitted parameters with respect to the true hit on
           the same surface, for the matched tracks
    @param tracks : DataFrame of tracks (analyse_event)
    @return the matched tracks with res_phi, res_theta (rad) and rel_pt (relative pT)
    '''
    t = tracks[tracks["category"] == "matched"].dropna(subset=["tpx"]).copy()
    p_true = np.sqrt(t["tpx"] ** 2 + t["tpy"] ** 2 + t["tpz"] ** 2)
    phi_true = np.arctan2(t["tpy"], t["tpx"])
    theta_true = np.arccos(np.clip(t["tpz"] / p_true, -1, 1))
    t["res_phi"] = np.angle(np.exp(1j * (t["phi"] - phi_true)))   # wrapped to [-pi, pi]
    t["res_theta"] = t["theta"] - theta_true
    pt_reco = np.sin(t["theta"]) / np.abs(t["qop"])
    pt_true = np.hypot(t["tpx"], t["tpy"])
    t["rel_pt"] = (pt_reco - pt_true) / pt_true
    t["pt_hit"] = pt_true
    return t


## Parameters of the pulls: column, axis label
PULLS = (("pull_phi", r"$\phi$"), ("pull_theta", r"$\theta$"), ("pull_qop", r"$q/p$"),
         ("pull_pt", r"$p_T$"))

## Covariance columns needed by the pulls (pT also needs theta - q/p)
COV_COLUMNS = ("cov_phi_phi", "cov_theta_theta", "cov_qop_qop", "cov_theta_qop")


def has_covariance(tracks):
    '''
    @brief tells whether the tracks have the covariance of the fit
    @param tracks : DataFrame of tracks
    @return (bool) True if the covariance columns used by the pulls are there
    '''
    return set(COV_COLUMNS).issubset(tracks.columns)


def robust_width(values):
    '''
    @brief width of a distribution that the tails do not inflate: half of the
           16-84% interval (equal to sigma for a Gaussian)
    @param values : array-like of numbers
    @return (float) the width, NaN without values
    '''
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    low, high = np.percentile(values, [16, 84])
    return 0.5 * (high - low)


def add_pulls(t):
    '''
    @brief pulls of the fitted parameters, from the covariance of the fit
    @param t : matched tracks with residuals (add_residuals) and the columns
           cov_<a>_<b>
    @return t with pull_phi, pull_theta, pull_qop and pull_pt; tracks with a
            non positive variance get NaN
    '''
    t = t.copy()
    p_true = np.sqrt(t["tpx"] ** 2 + t["tpy"] ** 2 + t["tpz"] ** 2)
    t["res_qop"] = t["qop"] - t["q"] / p_true

    def error(name):
        var = t[f"cov_{name}_{name}"]
        return np.sqrt(var.where(var > 0))

    t["pull_phi"] = t["res_phi"] / error("phi")
    t["pull_theta"] = t["res_theta"] / error("theta")
    t["pull_qop"] = t["res_qop"] / error("qop")

    # pT = sin(theta) / |q/p|: error propagated with the theta - q/p block
    abs_qop = np.abs(t["qop"])
    d_theta = np.cos(t["theta"]) / abs_qop
    d_qop = -np.sin(t["theta"]) * np.sign(t["qop"]) / t["qop"] ** 2
    var_pt = (d_theta ** 2 * t["cov_theta_theta"] + d_qop ** 2 * t["cov_qop_qop"]
              + 2 * d_theta * d_qop * t["cov_theta_qop"])
    pt_reco = np.sin(t["theta"]) / abs_qop
    t["pull_pt"] = (pt_reco - t["pt_hit"]) / np.sqrt(var_pt.where(var_pt > 0))

    return t


def analyse_run(tracks_folder, truth_folder, args):
    '''
    @brief analyses all the events of one run
    @param tracks_folder : (Path) folder <backend>_run_<N>_tracks
    @param truth_folder : (Path) simulated run folder
    @param args : command line options
    @return (tracks, particles) of all the events
    '''
    events = sorted(int(m.group(1)) for p in tracks_folder.glob("event*-tracks.csv")
                    if (m := EVENT_FILE.fullmatch(p.name)))
    results = [analyse_event(tracks_folder, truth_folder, e, args)
               for e in tqdm(events, desc=f"{tracks_folder.name:24s}")]
    if not results:
        raise SystemExit(f"No tracks in {tracks_folder}")
    tracks = pd.concat([r[0] for r in results], ignore_index=True)
    particles = pd.concat([r[1] for r in results], ignore_index=True)
    return tracks, particles


def summarize(tracks, particles):
    '''
    @brief efficiency, fake and duplicate rates and fit summary of one run
    @param tracks : DataFrame of the tracks of the run
    @param particles : DataFrame of the particles of the run
    @return (dict) the numbers, with binomial errors on the rates
    '''
    rec = particles[particles["reconstructable"]]
    n_rec, n_found = len(rec), int(rec["found"].sum())
    n_tracks = len(tracks)
    n_events = particles["event"].nunique() if "event" in particles else 1
    # fit_outcome = 1 is traccc's track_fit_outcome::SUCCESS; a file without
    # the column (e.g. the Hough tracks) counts every track as fitted
    if "fit_outcome" in tracks:
        n_fit_ok = int((tracks["fit_outcome"] == 1).sum())
    else:
        n_fit_ok = n_tracks

    def rate(k, n):
        return (k / n, np.sqrt(k / n * (1 - k / n) / n)) if n else (np.nan, np.nan)

    eff, eff_err = rate(n_found, n_rec)
    fake, fake_err = rate(int((tracks["category"] == "fake").sum()), n_tracks)
    dup, dup_err = rate(int((tracks["category"] == "duplicate").sum()), n_tracks)
    return {"events": n_events, "particles_total": len(particles),
            "charged": int((particles["q"] != 0).sum()) if "q" in particles else len(particles),
            "true_measurements": (int(particles["n_true_meas"].sum())
                                  if "n_true_meas" in particles else 0),
            "reco_measurements": int(tracks["n_meas"].sum()),
            "fit_success": n_fit_ok, "fit_failed": n_tracks - n_fit_ok,
            "reconstructable": n_rec, "found": n_found, "tracks": n_tracks,
            "efficiency": eff, "efficiency_err": eff_err,
            "fake_rate": fake, "fake_rate_err": fake_err,
            "duplicate_rate": dup, "duplicate_rate_err": dup_err,
            "mean_measurements": tracks["n_meas"].mean(),
            # Sanity check of the matching: reconstructed measurements with a
            # true measurement within the tolerance (should be close to 1)
            "measurement_match_fraction": (tracks["n_meas_matched"].sum() / tracks["n_meas"].sum()
                                           if tracks["n_meas"].sum() > 0 else np.nan),
            "mean_purity": tracks.loc[tracks["matched"], "purity"].mean()}


def legend_text(label, stats=""):
    '''
    @brief legend entry of a run: its name and its statistics; with a single run
           the name is empty (it is in the table of the first page) and only the
           statistics are left
    @param label : (str) name of the run, empty for a single run
    @param stats : (str) statistics of the run, e.g. "RMS 0.01"
    @return (str) the legend entry, or None for no entry
    '''
    text = ": ".join(part for part in (label, stats) if part)
    return text or None


def finish_legend(ax, fontsize=7):
    '''
    @brief draws the legend only if some line has an entry
    @param ax : matplotlib axes
    @param fontsize : font size of the legend
    '''
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=fontsize)


def binned_efficiency(particles, column, edges):
    '''
    @brief efficiency in bins of a particle variable
    @param particles : DataFrame of particles, with reconstructable and found
    @param column : (str) variable, e.g. "pt" or "eta"
    @param edges : bin edges
    @return (centres, efficiency, error) arrays; empty bins give NaN
    '''
    rec = particles[particles["reconstructable"]]
    total, _ = np.histogram(rec[column], edges)
    found, _ = np.histogram(rec.loc[rec["found"], column], edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = found / total
        err = np.sqrt(eff * (1 - eff) / total)
    return 0.5 * (edges[1:] + edges[:-1]), eff, err


def plot_rates(summary):
    '''
    @brief efficiency, fake and duplicate rates as a stem plot with binomial
           error bars: one stem per rate for a single run, one stem per run
           (against the particles per event) for several runs
    @param summary : DataFrame, one row per run (with particles and the rates)
    @return the matplotlib figure
    '''
    rates = (("efficiency", "Efficiency"), ("fake_rate", "Fake rate"),
             ("duplicate_rate", "Duplicate rate"))
    fig, ax = plt.subplots(figsize=(7, 5))

    if len(summary) == 1:
        # One run: the three rates side by side, with their values
        # The names are on the x axis: one colour and no legend
        row = summary.iloc[0]
        x = np.arange(len(rates))
        for i, (col, _) in enumerate(rates):
            ax.stem([x[i]], [row[col]], linefmt="C0-", markerfmt="C0o", basefmt=" ")
            ax.errorbar([x[i]], [row[col]], yerr=[row[f"{col}_err"]], fmt="none",
                        ecolor="C0", capsize=4)
            ax.annotate(f"{row[col]:.3f}", (x[i], row[col]), textcoords="offset points",
                        xytext=(8, 4), fontsize=9)
        ax.set_xticks(x, [label for _, label in rates])
        ax.set_xlim(-0.6, len(rates) - 0.4)
        ax.set_title("Tracking performance")
    else:
        # Several runs: stems against the particles per event, the three rates
        # slightly shifted (on the log axis) so that they do not overlap
        series_list = list(dict.fromkeys(summary["series"]))
        for i, (col, label) in enumerate(rates):
            for j, series in enumerate(series_list):
                s = summary[summary["series"] == series]
                shift = 1.12 ** (i - 1 + 3 * j)
                name = label if len(series_list) == 1 else f"{label} ({series})"
                colour = f"C{i + 3 * j}"
                ax.stem(s["particles"] * shift, s[col], linefmt=f"{colour}-",
                        markerfmt=f"{colour}o", basefmt=" ", label=name)
                ax.errorbar(s["particles"] * shift, s[col], yerr=s[f"{col}_err"],
                            fmt="none", ecolor=colour, capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("Particles per event")
        ax.set_title("Tracking performance against the event size")
        ax.grid(True, which="both", alpha=0.3)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    if len(summary) > 1:
        # Several runs: the colours tell the rates apart
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


## Rows of the table of the first page: summary column, row name, format.
## A format "rate" prints the value with its binomial error
TABLE_ROWS = (("events", "Events", "d"),
              ("particles", "Particles per event (simulated)", "d"),
              ("true_measurements", "True measurements", "d"),
              ("tracks", "Reconstructed tracks", "d"),
              ("reco_measurements", "Measurements on the tracks", "d"),
              ("mean_measurements", "Measurements per track", ".2f"),
              ("fit_success", "Successful fits", "d"),
              ("fit_failed", "Failed fits", "d"),
              ("mean_purity", "Mean purity of the matched tracks", ".3f"),
              ("measurement_match_fraction", "Measurements matched to the truth", ".3f"))


def table_cell(row, column, fmt):
    '''
    @brief text of one cell of the summary table
    @param row : one row of the summary (a run)
    @param column : (str) summary column
    @param fmt : (str) format, or "rate" for value +- error
    @return (str) the text, "-" if the value is missing
    '''
    value = row.get(column, np.nan)
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if fmt == "rate":
        return rf"{value:.3f} $\pm$ {row[column + '_err']:.3f}"
    if fmt == "d":
        return f"{int(value):d}"
    return format(value, fmt)


def plot_summary_table(summary, args):
    '''
    @brief first page: the characteristics of every run (particles, measurements,
           tracks, fits) and its rates, one column per run, with the cuts of the
           analysis below
    @param summary : DataFrame, one row per run
    @param args : command line options (tolerance_mm, min_fraction, min_pt, min_hits)
    @return the matplotlib figure
    '''
    single_series = summary["series"].nunique() == 1
    headers = [r["run"] if single_series else f"{r['series']}\n{r['run']}"
               for _, r in summary.iterrows()]
    cells = [[table_cell(r, column, fmt) for _, r in summary.iterrows()]
             for column, _, fmt in TABLE_ROWS]
    names = [name for _, name, _ in TABLE_ROWS]

    width = 4.5 + 1.6 * len(headers)
    fig, ax = plt.subplots(figsize=(max(width, 7), 0.3 * len(TABLE_ROWS) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=cells, rowLabels=names, colLabels=headers,
                     cellLoc="center", rowLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        if i == 0 or j == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.93")
    title = "Reconstructed runs"
    if single_series:
        title += f" ({summary['series'].iloc[0]})"
    ax.set_title(title, fontsize=12, pad=12)
    cuts = (rf"Matching: tolerance {args.tolerance_mm:g} mm, track matched with "
            rf"$\geq${args.min_fraction:.0%} of its measurements from one particle.  "
            "\n" rf"Reconstructable: charged, $p_T > {args.min_pt:g}$ GeV, "
            rf"$\geq${args.min_hits} true measurements.  "
            "Successful fit: fit_outcome = 1 (traccc SUCCESS).")
    ax.text(0.5, -0.02, cuts, transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="0.3")
    return fig


def zoomed_limits(low, high, min_span=0.04, max_value=1.0):
    '''
    @brief y range of an efficiency plot around its points: the efficiency is
           expected flat, so the full 0-1 range would hide its variations
    @param low, high : lowest and highest point (with its error) of every run
    @param min_span : (float) smallest range shown
    @param max_value : (float) highest value shown, plus a margin
    @return (bottom, top) of the y axis
    '''
    low = np.nanmin(low) if np.isfinite(low).any() else 0.0
    high = np.nanmax(high) if np.isfinite(high).any() else 1.0
    margin = 0.15 * max(high - low, min_span)
    top = min(high + margin, max_value + 0.2 * margin)
    bottom = max(min(low - margin, top - min_span), 0.0)
    return bottom, top


def plot_binned(runs, column, edges, xlabel, name, log_x=False):
    '''
    @brief efficiency against a particle variable, one line per run, with the
           integrated efficiency of the run as a dashed line; the y axis is
           zoomed around the points
    @param runs : list of (label, particles) pairs
    @param column, edges, xlabel : variable, bins and axis label
    @param name : (str) name of the variable in the title, e.g. r"$\\eta$"
    @param log_x : logarithmic x axis
    @return the matplotlib figure
    '''
    fig, ax = plt.subplots(figsize=(7, 5))
    low, high = [], []
    for label, particles in runs:
        x, eff, err = binned_efficiency(particles, column, edges)
        bars = ax.errorbar(x, eff, yerr=err, marker="o", ms=3, capsize=2,
                           label=legend_text(label))
        # Integrated efficiency of the run, as a reference for the bins
        rec = particles[particles["reconstructable"]]
        if len(rec):
            ax.axhline(rec["found"].mean(), color=bars[0].get_color(), lw=0.8, ls="--",
                       label=legend_text(label, f"integrated {rec['found'].mean():.3f}"))
        good = np.isfinite(eff)
        low.append(np.min((eff - err)[good]) if good.any() else np.nan)
        high.append(np.max((eff + err)[good]) if good.any() else np.nan)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(*zoomed_limits(low, high))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Efficiency")
    ax.set_title(f"Efficiency against {name}")
    ax.grid(True, which="both", alpha=0.3)
    finish_legend(ax)
    fig.tight_layout()
    return fig


## Residuals of the fitted parameters: column, axis label, title
RESIDUALS = (("rel_pt",
              r"$(p_T^{\mathrm{reco}} - p_T^{\mathrm{true}})\,/\,p_T^{\mathrm{true}}$",
              r"Relative $p_T$ resolution"),
             ("res_phi", r"$\phi_{\mathrm{reco}} - \phi_{\mathrm{true}}$ [rad]",
              r"$\phi$ resolution"),
             ("res_theta", r"$\theta_{\mathrm{reco}} - \theta_{\mathrm{true}}$ [rad]",
              r"$\theta$ resolution"))


def plot_residual(runs, column, xlabel, title):
    '''
    @brief distribution of the residual of one fitted parameter, one line per
           run, with the RMS of every run in the legend
    @param runs : list of (label, matched tracks with residuals) pairs
    @param column : (str) residual column, e.g. "rel_pt"
    @param xlabel, title : axis label and title
    @return the matplotlib figure, or None if there are no residuals
    '''
    values = pd.concat([t[column] for _, t in runs]).dropna()
    if values.empty:
        return None

    low, high = np.percentile(values, [1, 99])
    edges = np.linspace(low, high, 61)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, t in runs:
        v = t[column].dropna()
        inside = v[(v >= low) & (v <= high)]
        rms = np.sqrt(np.mean(inside ** 2)) if len(inside) else np.nan
        ax.hist(v, bins=edges, histtype="step", linewidth=1.5, density=True,
                label=legend_text(label, f"RMS {rms:.3g}, {len(v)} tracks"))
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("normalized tracks")
    ax.set_title(f"{title} (matched tracks)")
    finish_legend(ax)
    fig.tight_layout()
    return fig


def plot_pvalue(runs):
    '''
    @brief p-value of the fit, flat for a correct fit
    @param runs : list of (label, tracks) pairs
    @return the matplotlib figure, or None if no track has a p-value (e.g. the
            Hough fit, ndf = 0)
    '''
    good_runs = [(label, t[(t["ndf"] > 0) & t["pval"].notna()]) for label, t in runs]
    good_runs = [(label, t) for label, t in good_runs if len(t)]
    if not good_runs:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, t in good_runs:
        ax.hist(t["pval"], bins=np.linspace(0, 1, 21), histtype="step", linewidth=1.5,
                density=True, label=legend_text(label))
    ax.axhline(1, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("fit p-value")
    ax.set_ylabel("normalized tracks")
    ax.set_title("Fit p-value (flat for a correct fit)")
    finish_legend(ax)
    fig.tight_layout()
    return fig


def plot_track_length(runs):
    '''
    @brief measurements per track
    @param runs : list of (label, tracks) pairs
    @return the matplotlib figure
    '''
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, t in runs:
        n = t["n_meas"].dropna().astype(int)
        if len(n):
            ax.hist(n, bins=np.arange(n.min(), n.max() + 2) - 0.5, histtype="step",
                    linewidth=1.5, density=True, label=legend_text(label))
    ax.set_xlabel("measurements per track")
    ax.set_ylabel("normalized tracks")
    ax.set_title("Track length")
    finish_legend(ax)
    fig.tight_layout()
    return fig


def plot_pulls(runs):
    '''
    @brief pulls of phi, theta, q/p and pT, one panel per parameter and one
           line per run, with the unit Gaussian they must follow
    @param runs : list of (label, tracks with pulls) pairs
    @return the matplotlib figure
    '''
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    edges = np.linspace(-5, 5, 51)
    x = np.linspace(-5, 5, 201)
    for ax, (column, name) in zip(axes.flat, PULLS):
        for label, t in runs:
            v = t[column].dropna()
            if v.empty:
                continue
            ax.hist(v, bins=edges, histtype="step", linewidth=1.5, density=True,
                    label=legend_text(label, f"median {v.median():.2f}, "
                                                     f"width {robust_width(v):.2f}"))
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8,
                label=r"$\mathcal{N}(0,1)$")
        ax.set_xlabel(rf"pull of {name} = (fit $-$ true) / $\sigma$")
        ax.set_ylabel("normalized tracks")
        ax.legend(fontsize=6)
    fig.suptitle("Pulls of the fitted parameters (matched tracks); "
                 "median and half of the 16-84% interval in the legend")
    fig.tight_layout()
    return fig


def find_runs(folders, data_root):
    '''
    @brief finds the track folders and their truth folders
    @param folders : list of results folders (searched with their subfolders)
    @param data_root : (Path) folder with the simulated runs
    @return list of (series, run name, particles, tracks folder, truth folder),
            sorted by series and number of particles
    '''
    found = []
    for folder in map(Path, folders):
        candidates = [folder] + sorted(folder.rglob("*_tracks"))
        for path in candidates:
            match = TRACKS_FOLDER.fullmatch(path.name)
            if not (path.is_dir() and match):
                continue
            truth = data_root / match["run"]
            if not truth.is_dir():
                print(f"Skipped {path}: no truth folder {truth}")
                continue
            series = path.parent.name if path.parent != folder.parent else folder.name
            found.append((series, match["run"], int(match["n"]), path, truth))
    if not found:
        raise SystemExit("No <backend>_run_<N>_tracks folders found (see exec/run_reco.sh)")
    return sorted(found, key=lambda r: (r[0], r[2]))


def main():
    parser = argparse.ArgumentParser(description="Efficiency, fake rate and resolution "
                                                 "of the reconstructed tracks.")
    parser.add_argument("folders", nargs="*",
                        help="results folders with <backend>_run_<N>_tracks subfolders")
    parser.add_argument("--data", type=Path,
                        help="folder with the simulated runs (run_<N>), e.g. data/run_100ev")
    parser.add_argument("--tracks", type=Path,
                        help="single run: folder with its tracks, e.g. data/reco_gpu")
    parser.add_argument("--truth", type=Path,
                        help="single run: its simulated run folder, e.g. data/run_100ev/run_500")
    parser.add_argument("--tolerance-mm", type=float, default=0.25,
                        help="max distance between reconstructed and true measurement (mm)")
    parser.add_argument("--min-fraction", type=float, default=0.5,
                        help="fraction of the measurements of a track from one particle "
                             "to match it (default 0.5)")
    parser.add_argument("--min-pt", type=float, default=0.6,
                        help="minimum pT of a reconstructable particle in GeV (default 0.6)")
    parser.add_argument("--min-hits", type=int, default=3,
                        help="minimum true measurements of a reconstructable particle")
    parser.add_argument("-n", "--name", default="physics_performance", help="name of the PDF")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"),
                        help="output folder (default: results)")
    args = parser.parse_args()

    if args.tracks or args.truth:
        if not (args.tracks and args.truth) or args.folders:
            parser.error("a single run needs --tracks and --truth, and no other folder")
        for folder in (args.tracks, args.truth):
            if not folder.is_dir():
                parser.error(f"folder not found: {folder}")
        n = int(m.group(1)) if (m := re.search(r"(\d+)$", args.truth.name)) else 0
        runs = [(args.tracks.name, args.truth.name, n, args.tracks, args.truth)]
    else:
        if not args.folders or args.data is None:
            parser.error("give --tracks and --truth, or results folders and --data")
        runs = find_runs(args.folders, args.data)
    rows, eff_runs, res_runs, fit_runs = [], [], [], []
    for series, run, n, tracks_folder, truth in runs:
        tracks, particles = analyse_run(tracks_folder, truth, args)
        # The run names are in the table of the first page: a single run needs
        # no name in the legends
        if len(runs) == 1:
            label = ""
        elif len({r[0] for r in runs}) == 1:
            label = run
        else:
            label = f"{series} {run}"
        rows.append({"series": series, "run": run, "particles": n,
                     **summarize(tracks, particles)})
        eff_runs.append((label, particles))
        residuals = add_residuals(tracks)
        if has_covariance(residuals):
            residuals = add_pulls(residuals)
        res_runs.append((label, residuals))
        fit_runs.append((label, tracks))

    summary = pd.DataFrame(rows)
    pt_edges = np.geomspace(0.1, 100, 31)
    eta_edges = np.linspace(-4, 4, 33)
    figs = [plot_summary_table(summary, args),
            plot_rates(summary),
            plot_binned(eff_runs, "pt", pt_edges, r"$p_T$ [GeV]", r"$p_T$", log_x=True),
            plot_binned(eff_runs, "eta", eta_edges, r"$\eta$", r"$\eta$")]
    # One page per residual
    for column, xlabel, title in RESIDUALS:
        fig = plot_residual(res_runs, column, xlabel, title)
        if fig is not None:
            figs.append(fig)
    fig = plot_pvalue(fit_runs)
    if fig is not None:
        figs.append(fig)
    figs.append(plot_track_length(fit_runs))

    # Errors of the fit, for the runs with the covariance in their tracks file
    pull_runs = [(label, t) for label, t in res_runs if "pull_phi" in t]
    if pull_runs:
        figs.append(plot_pulls(pull_runs))
        for label, t in pull_runs:
            widths = ", ".join(f"{col[5:]} {robust_width(t[col]):.2f}" for col, _ in PULLS)
            print(f"{label or runs[0][1]}: width of the pulls "
                  f"(1 if the errors are right): {widths}")
    else:
        print("No covariance in the tracks files: the pulls are not drawn")

    args.output.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output / f"{args.name}.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
    summary.to_csv(pdf_path.with_suffix(".csv"), index=False, float_format="%.4g")
    print(summary[["series", "run", "reconstructable", "tracks", "efficiency",
                   "fake_rate", "duplicate_rate", "measurement_match_fraction"]]
          .to_string(index=False))
    print(f"{len(figs)} plots saved in {pdf_path}, table in {pdf_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()