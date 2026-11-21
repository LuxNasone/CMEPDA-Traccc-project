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
The track parameters are given on the surface of the first measurement of the
track, so they are compared with the true hit on that surface.

If the tracks file has the covariance of the parameters (columns cov_<a>_<b>,
written by csv_writer.hpp), the errors of the fit are checked too:
  - pulls of phi, theta, q/p and pT, (fit - true) / error, which must be
    Gaussian with mean 0 and width 1 if the errors are right;
  - their width against the true pT (multiple scattering grows at low pT);
  - the chi2 of (phi, theta, q/p) with the full 3x3 covariance, which must
    follow a chi2 with 3 degrees of freedom (flat p-value).

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
import math
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
PULLS = (("pull_phi", "phi"), ("pull_theta", "theta"), ("pull_qop", "q/p"), ("pull_pt", "pT"))

## Parameters of the chi2 test, in the order of the covariance columns
CHI2_PARAMS = ("phi", "theta", "qop")


def has_covariance(tracks):
    '''
    @brief tells whether the tracks have the covariance of the fit
    @param tracks : DataFrame of tracks
    @return (bool) True if the columns of the (phi, theta, qop) block are there
    '''
    names = {f"cov_{a}_{b}" for i, a in enumerate(CHI2_PARAMS) for b in CHI2_PARAMS[i:]}
    return names.issubset(tracks.columns)


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


def chi2_3_sf(x):
    '''
    @brief survival function (p-value) of a chi2 with 3 degrees of freedom
    @param x : array of chi2 values (>= 0)
    @return array of p-values
    '''
    x = np.asarray(x, dtype=float)
    erfc = np.vectorize(math.erfc, otypes=[float])
    return erfc(np.sqrt(x / 2)) + np.sqrt(2 * x / np.pi) * np.exp(-x / 2)


def add_pulls(t):
    '''
    @brief pulls of the fitted parameters and chi2 of (phi, theta, q/p), from
           the covariance of the fit
    @param t : matched tracks with residuals (add_residuals) and the columns
           cov_<a>_<b>
    @return t with pull_phi, pull_theta, pull_qop, pull_pt, chi2_3 and
            chi2_3_pval; tracks with a non positive variance get NaN
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

    # chi2 = r^T C^-1 r with the 3x3 covariance of (phi, theta, q/p)
    n = len(t)
    cov = np.empty((n, 3, 3))
    for i, a in enumerate(CHI2_PARAMS):
        for j, b in enumerate(CHI2_PARAMS):
            name = f"cov_{a}_{b}" if i <= j else f"cov_{b}_{a}"
            cov[:, i, j] = t[name].to_numpy(dtype=float)
    res = t[["res_phi", "res_theta", "res_qop"]].to_numpy(dtype=float)
    chi2 = np.full(n, np.nan)
    good = np.isfinite(cov).all(axis=(1, 2)) & np.isfinite(res).all(axis=1)
    if good.any():
        # Only positive definite matrices: the others come from failed fits
        positive = np.zeros(n, dtype=bool)
        positive[good] = np.linalg.eigvalsh(cov[good]).min(axis=1) > 0
        if positive.any():
            solved = np.linalg.solve(cov[positive], res[positive][..., None])[..., 0]
            chi2[positive] = np.einsum("ij,ij->i", res[positive], solved)
    t["chi2_3"] = chi2
    t["chi2_3_pval"] = np.where(np.isfinite(chi2), chi2_3_sf(np.nan_to_num(chi2)), np.nan)
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

    def rate(k, n):
        return (k / n, np.sqrt(k / n * (1 - k / n) / n)) if n else (np.nan, np.nan)

    eff, eff_err = rate(n_found, n_rec)
    fake, fake_err = rate(int((tracks["category"] == "fake").sum()), n_tracks)
    dup, dup_err = rate(int((tracks["category"] == "duplicate").sum()), n_tracks)
    return {"reconstructable": n_rec, "found": n_found, "tracks": n_tracks,
            "efficiency": eff, "efficiency_err": eff_err,
            "fake_rate": fake, "fake_rate_err": fake_err,
            "duplicate_rate": dup, "duplicate_rate_err": dup_err,
            "mean_measurements": tracks["n_meas"].mean(),
            # Sanity check of the matching: reconstructed measurements with a
            # true measurement within the tolerance (should be close to 1)
            "measurement_match_fraction": (tracks["n_meas_matched"].sum() / tracks["n_meas"].sum()
                                           if tracks["n_meas"].sum() > 0 else np.nan),
            "mean_purity": tracks.loc[tracks["matched"], "purity"].mean()}


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
        row = summary.iloc[0]
        x = np.arange(len(rates))
        for i, (col, label) in enumerate(rates):
            ax.stem([x[i]], [row[col]], linefmt=f"C{i}-", markerfmt=f"C{i}o",
                    basefmt=" ", label=label)
            ax.errorbar([x[i]], [row[col]], yerr=[row[f"{col}_err"]], fmt="none",
                        ecolor=f"C{i}", capsize=4)
            ax.annotate(f"{row[col]:.3f}", (x[i], row[col]), textcoords="offset points",
                        xytext=(8, 4), fontsize=9)
        ax.set_xticks(x, [label for _, label in rates])
        ax.set_xlim(-0.6, len(rates) - 0.4)
        ax.set_title(f"Tracking performance, {row['run']} "
                     f"({int(row['reconstructable'])} reconstructable particles, "
                     f"{int(row['tracks'])} tracks)", fontsize=10)
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
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_binned(runs, column, edges, xlabel, log_x=False):
    '''
    @brief efficiency against a particle variable, one line per run
    @param runs : list of (label, particles) pairs
    @param column, edges, xlabel : variable, bins and axis label
    @param log_x : logarithmic x axis
    @return the matplotlib figure
    '''
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, particles in runs:
        x, eff, err = binned_efficiency(particles, column, edges)
        ax.errorbar(x, eff, yerr=err, marker="o", ms=3, capsize=2, label=label)
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Efficiency")
    ax.set_title(f"Efficiency against {xlabel.split(' [')[0]}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


## Residuals of the fitted parameters: column, axis label, title
RESIDUALS = (("rel_pt", "(pT reco - pT true) / pT true", "Relative pT resolution"),
             ("res_phi", "phi reco - phi true [rad]", "phi resolution"),
             ("res_theta", "theta reco - theta true [rad]", "theta resolution"))


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
    # Range from the central 98% of all the runs, so that the outliers do not
    # squeeze the peak; the RMS is computed in the same range
    low, high = np.percentile(values, [1, 99])
    edges = np.linspace(low, high, 61)
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, t in runs:
        v = t[column].dropna()
        inside = v[(v >= low) & (v <= high)]
        rms = np.sqrt(np.mean(inside ** 2)) if len(inside) else np.nan
        ax.hist(v, bins=edges, histtype="step", linewidth=1.5, density=True,
                label=f"{label}: RMS {rms:.3g}, {len(v)} tracks")
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("normalized tracks")
    ax.set_title(f"{title} (matched tracks)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_fit_quality(runs):
    '''
    @brief p-value of the fit (flat for a correct fit) and measurements per track
    @param runs : list of (label, tracks) pairs
    @return the matplotlib figure
    '''
    fig, (ax_p, ax_n) = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, t in runs:
        good = t[(t["ndf"] > 0) & t["pval"].notna()]
        # Tracks without a p-value (e.g. the Hough fit, ndf = 0) are not drawn
        if len(good):
            ax_p.hist(good["pval"], bins=np.linspace(0, 1, 21), histtype="step",
                      density=True, label=label)
        n = t["n_meas"].dropna().astype(int)
        if len(n):
            ax_n.hist(n, bins=np.arange(n.min(), n.max() + 2) - 0.5, histtype="step",
                      density=True, label=label)
    ax_p.set_xlabel("fit p-value")
    ax_p.set_ylabel("normalized tracks")
    ax_p.set_title("Fit p-value (flat for a correct fit)")
    ax_n.set_xlabel("measurements per track")
    ax_n.set_ylabel("normalized tracks")
    ax_n.set_title("Track length")
    for ax in (ax_p, ax_n):
        ax.legend(fontsize=6)
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
                    label=f"{label}: mean {v.median():.2f}, width {robust_width(v):.2f}")
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8,
                label="N(0, 1)")
        ax.set_xlabel(f"pull of {name} = (fit - true) / error")
        ax.set_ylabel("normalized tracks")
        ax.legend(fontsize=6)
    fig.suptitle("Pulls of the fitted parameters (matched tracks); "
                 "median and half of the 16-84% interval in the legend")
    fig.tight_layout()
    return fig


def plot_pull_width(runs, edges):
    '''
    @brief width of the pulls against the true pT: 1 if the errors are right,
           a trend with pT points to multiple scattering or energy loss
    @param runs : list of (label, tracks with pulls) pairs
    @param edges : pT bin edges in GeV
    @return the matplotlib figure
    '''
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    centres = np.sqrt(edges[1:] * edges[:-1])
    for ax, (column, name) in zip(axes.flat, PULLS):
        for label, t in runs:
            bins = pd.cut(t["pt_hit"], edges)
            width = t.groupby(bins, observed=False)[column].agg(robust_width)
            ax.plot(centres, width.to_numpy(), marker="o", ms=3, label=label)
        ax.axhline(1, color="grey", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("true pT [GeV]")
        ax.set_ylabel(f"width of the pull of {name}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("Width of the pulls against pT (1 = correct errors)")
    fig.tight_layout()
    return fig


def plot_chi2_3(runs):
    '''
    @brief chi2 of (phi, theta, q/p) against the truth with the 3x3 covariance:
           it follows a chi2 with 3 degrees of freedom, and its p-value is flat,
           if the covariance (correlations included) is right
    @param runs : list of (label, tracks with chi2_3) pairs
    @return the matplotlib figure
    '''
    fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(12, 4.5))
    edges = np.linspace(0, 15, 61)
    x = np.linspace(1e-3, 15, 300)
    for label, t in runs:
        chi2 = t["chi2_3"].dropna()
        if chi2.empty:
            continue
        ax_c.hist(chi2.clip(upper=edges[-1]), bins=edges, histtype="step", density=True,
                  label=f"{label}: mean {chi2.mean():.2f} (3 expected)")
        ax_p.hist(t["chi2_3_pval"].dropna(), bins=np.linspace(0, 1, 21), histtype="step",
                  density=True, label=label)
    ax_c.plot(x, np.sqrt(x) * np.exp(-x / 2) / np.sqrt(2 * np.pi), "k--", lw=0.8,
              label="chi2, 3 d.o.f.")
    ax_p.axhline(1, color="k", ls="--", lw=0.8, label="flat")
    ax_c.set_xlabel("chi2 of (phi, theta, q/p) against the truth (last bin: overflow)")
    ax_p.set_xlabel("p-value of that chi2")
    for ax in (ax_c, ax_p):
        ax.set_ylabel("normalized tracks")
        ax.legend(fontsize=6)
    fig.suptitle("Covariance of the fit against the truth (matched tracks)")
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
        label = f"{run}" if len({r[0] for r in runs}) == 1 else f"{series} {run}"
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
    figs = [plot_rates(summary),
            plot_binned(eff_runs, "pt", pt_edges, "pT [GeV]", log_x=True),
            plot_binned(eff_runs, "eta", eta_edges, "eta")]
    # One page per residual
    for column, xlabel, title in RESIDUALS:
        fig = plot_residual(res_runs, column, xlabel, title)
        if fig is not None:
            figs.append(fig)
    figs.append(plot_fit_quality(fit_runs))

    # Errors of the fit, for the runs with the covariance in their tracks file
    pull_runs = [(label, t) for label, t in res_runs if "pull_phi" in t]
    if pull_runs:
        figs += [plot_pulls(pull_runs), plot_pull_width(pull_runs, pt_edges),
                 plot_chi2_3(pull_runs)]
        for label, t in pull_runs:
            widths = ", ".join(f"{name} {robust_width(t[col]):.2f}" for col, name in PULLS)
            print(f"{label}: width of the pulls: {widths}; "
                  f"mean chi2(phi, theta, q/p) {t['chi2_3'].mean():.2f} (3 expected)")
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