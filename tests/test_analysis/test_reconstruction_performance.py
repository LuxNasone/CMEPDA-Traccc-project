'''
@brief Tests of src/analysis/reconstruction_performance.py, on one small event
       whose expected efficiency, fake and duplicate rates are known.

The event (written by write_event) has five particles and four tracks:
  - particle 0: 4 hits, pT 2 GeV  -> track 0 (exact) and track 1 (its duplicate)
  - particle 1: 4 hits, pT 2 GeV  -> track 2, made of 2 of its hits and 1 of
                                     particle 2: purity 2/3, matched to it
  - particle 2: 4 hits, pT 2 GeV  -> no track of its own: lost
  - particle 3: 4 hits, pT 0.3 GeV -> below the pT cut, not reconstructable
  - particle 4: 2 hits, pT 2 GeV  -> too few hits, not reconstructable
  - track 3: measurements far from any true one: fake
So: 3 reconstructable particles, 2 found (efficiency 2/3), 4 tracks of which
1 fake and 1 duplicate (rates 1/4 and 1/4).
'''

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import reconstruction_performance as rp

## The script, to run it as a program
SCRIPT = Path(rp.__file__)

## Options of the analysis, as the command line defaults
ARGS = SimpleNamespace(tolerance_mm=0.25, min_fraction=0.5, min_pt=0.6, min_hits=3)

## Particles of the test event: id -> (pT in GeV, phi, number of hits)
PARTICLES = {0: (2.0, 0.5, 4), 1: (2.0, 1.5, 4), 2: (2.0, -1.0, 4),
             3: (0.3, 2.5, 4), 4: (2.0, -2.5, 2)}


def module_of(particle, layer):
    '''@brief geometry_id of the module crossed by a particle on a layer'''
    return 1000 * (layer + 1) + particle


def local_of(particle, layer):
    '''@brief true local position of a particle on a layer, in mm'''
    return 0.5 * particle + 0.1 * layer, -1.0 * particle + 0.2 * layer


def write_event(truth_folder, tracks_folder, event=0):
    '''
    @brief writes the truth and the tracks of the test event described above
    @param truth_folder : (Path) simulated run folder
    @param tracks_folder : (Path) folder of the reconstructed tracks
    @param event : (int) event number
    '''
    truth_folder.mkdir(parents=True, exist_ok=True)
    tracks_folder.mkdir(parents=True, exist_ok=True)
    particles, hits, meas, links = [], [], [], []
    for pid, (pt, phi, n_hits) in PARTICLES.items():
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), 0.5 * pt
        particles.append({"particle_id": pid, "particle_type": 13, "process": 0,
                          "vx": 0, "vy": 0, "vz": 0, "vt": 0,
                          "px": px, "py": py, "pz": pz, "m": 0, "q": -1})
        for layer in range(n_hits):
            l0, l1 = local_of(pid, layer)
            links.append({"measurement_id": len(meas), "hit_id": len(hits)})
            meas.append({"measurement_id": len(meas), "geometry_id": module_of(pid, layer),
                         "local_key": "", "local0": l0, "local1": l1, "phi": phi,
                         "theta": 0, "time": 0, "var_local0": 0, "var_local1": 0,
                         "var_phi": 0, "var_theta": 0, "var_time": 0})
            hits.append({"particle_id": pid, "geometry_id": module_of(pid, layer),
                         "tx": 0, "ty": 0, "tz": 0, "tt": 0, "tpx": px, "tpy": py,
                         "tpz": pz, "te": 0, "deltapx": 0, "deltapy": 0, "deltapz": 0,
                         "deltae": 0, "index": 0})

    # Track measurements: (track, particle, layer), the position a little off
    # the true one, as the centre of a cluster would be
    content = [(0, 0, layer) for layer in range(4)] + \
              [(1, 0, layer) for layer in range(4)] + \
              [(2, 1, 0), (2, 1, 1), (2, 2, 2)]
    track_meas = []
    for track, pid, layer in content:
        l0, l1 = local_of(pid, layer)
        track_meas.append({"track_id": track, "geometry_id": module_of(pid, layer),
                           "local0": l0 + 0.02, "local1": l1 - 0.01})
    for layer in range(3):   # fake track: no true measurement nearby
        track_meas.append({"track_id": 3, "geometry_id": module_of(0, layer),
                           "local0": 50.0, "local1": 50.0})

    # Fitted parameters: exact for tracks 0 and 2, so their residuals are zero
    def params(pid):
        pt, phi, _ = PARTICLES[pid]
        p = np.hypot(pt, 0.5 * pt)
        return {"phi": phi, "theta": np.arccos(0.5 * pt / p), "qop": -1 / p}

    tracks = []
    for track, pid in ((0, 0), (1, 0), (2, 1), (3, 0)):
        n = sum(1 for t, _, _ in content if t == track) or 3
        tracks.append({"track_id": track, "fit_outcome": 1, "chi2": 1.0,
                       "ndf": 2 * n - 5, "pval": 0.5, "n_measurements": n,
                       "n_holes": 0, "loc0": 0, "loc1": 0, **params(pid)})

    name = f"event{event:09d}"
    pd.DataFrame(particles).to_csv(truth_folder / f"{name}-particles_initial.csv", index=False)
    pd.DataFrame(hits).to_csv(truth_folder / f"{name}-hits.csv", index=False)
    pd.DataFrame(meas).to_csv(truth_folder / f"{name}-measurements.csv", index=False)
    pd.DataFrame(links).to_csv(truth_folder / f"{name}-measurement-simhit-map.csv",
                               index=False)
    pd.DataFrame(tracks).to_csv(tracks_folder / f"{name}-tracks.csv", index=False)
    pd.DataFrame(track_meas).to_csv(tracks_folder / f"{name}-track-measurements.csv",
                                    index=False)


@pytest.fixture
def event(tmp_path):
    '''@brief the test event analysed: (tracks, particles)'''
    write_event(tmp_path / "run_5", tmp_path / "tracks")
    return rp.analyse_event(tmp_path / "tracks", tmp_path / "run_5", 0, ARGS)


# --- truth -----------------------------------------------------------------

def test_read_truth_links_measurements_to_particles(tmp_path):
    write_event(tmp_path / "run_5", tmp_path / "tracks")
    meas, particles = rp.read_truth(tmp_path / "run_5", 0)
    assert len(meas) == sum(n for _, _, n in PARTICLES.values())
    # Every measurement goes back to the particle that crossed its module
    assert (meas["geometry_id"] % 1000 == meas["particle_id"]).all()
    particles = particles.set_index("particle_id")
    assert particles.loc[0, "pt"] == pytest.approx(2.0)
    assert particles.loc[4, "n_true_meas"] == 2


# --- matching of the measurements ------------------------------------------

def test_match_measurements_takes_the_closest_on_the_same_module():
    truth = pd.DataFrame({"geometry_id": [1, 1, 2], "local0": [0.0, 1.0, 0.0],
                          "local1": [0.0, 0.0, 0.0], "particle_id": [10, 11, 12],
                          "tpx": 1.0, "tpy": 0.0, "tpz": 0.0})
    reco = pd.DataFrame({"track_id": [0, 0, 0], "geometry_id": [1, 1, 3],
                         "local0": [0.9, 0.1, 0.0], "local1": [0.0, 0.0, 0.0]})
    matched = rp.match_measurements(reco, truth, tolerance=0.25)
    assert list(matched["particle_id"][:2]) == [11, 10]
    assert np.isnan(matched["particle_id"][2])          # no true measurement on module 3


def test_match_measurements_respects_the_tolerance():
    truth = pd.DataFrame({"geometry_id": [1], "local0": [0.0], "local1": [0.0],
                          "particle_id": [10], "tpx": 1.0, "tpy": 0.0, "tpz": 0.0})
    reco = pd.DataFrame({"track_id": [0], "geometry_id": [1],
                         "local0": [0.3], "local1": [0.0]})
    assert np.isnan(rp.match_measurements(reco, truth, 0.25)["particle_id"][0])
    assert rp.match_measurements(reco, truth, 0.5)["particle_id"][0] == 10


# --- one event ---------------------------------------------------------------

def test_track_categories(event):
    tracks, _ = event
    categories = tracks.set_index("track_id")["category"].to_dict()
    assert categories == {0: "matched", 1: "duplicate", 2: "matched", 3: "fake"}


def test_purity_of_a_mixed_track(event):
    tracks, _ = event
    track = tracks.set_index("track_id").loc[2]
    assert track["particle_id"] == 1
    assert track["purity"] == pytest.approx(2 / 3)


def test_reconstructable_and_found(event):
    _, particles = event
    particles = particles.set_index("particle_id")
    assert list(particles["reconstructable"]) == [True, True, True, False, False]
    assert list(particles["found"]) == [True, True, False, False, False]


def test_stricter_purity_turns_the_mixed_track_into_a_fake(tmp_path):
    write_event(tmp_path / "run_5", tmp_path / "tracks")
    strict = SimpleNamespace(**{**vars(ARGS), "min_fraction": 0.8})
    tracks, _ = rp.analyse_event(tmp_path / "tracks", tmp_path / "run_5", 0, strict)
    assert tracks.set_index("track_id").loc[2, "category"] == "fake"


def test_summarize(event):
    tracks, particles = event
    row = rp.summarize(tracks, particles)
    assert (row["reconstructable"], row["found"], row["tracks"]) == (3, 2, 4)
    assert row["efficiency"] == pytest.approx(2 / 3)
    assert row["fake_rate"] == pytest.approx(1 / 4)
    assert row["duplicate_rate"] == pytest.approx(1 / 4)
    assert row["efficiency_err"] == pytest.approx(np.sqrt(2 / 3 * 1 / 3 / 3))
    # Fake track: 3 measurements without a true one; the others all matched
    assert row["measurement_match_fraction"] == pytest.approx(11 / 14)


def test_summarize_without_reconstructable_particles():
    particles = pd.DataFrame({"reconstructable": [False], "found": [False]})
    tracks = pd.DataFrame({"category": [], "n_meas": [], "n_meas_matched": [],
                           "matched": [], "purity": []})
    row = rp.summarize(tracks, particles)
    assert np.isnan(row["efficiency"]) and np.isnan(row["fake_rate"])
    assert np.isnan(row["measurement_match_fraction"])


# --- residuals and binned efficiency -----------------------------------------

def test_exact_parameters_give_zero_residuals(event):
    tracks, _ = event
    residuals = rp.add_residuals(tracks)
    assert sorted(residuals["track_id"]) == [0, 2]      # matched tracks only
    for column in ("res_phi", "res_theta", "rel_pt"):
        assert residuals[column].abs().max() < 1e-9


def test_phi_residual_is_wrapped():
    tracks = pd.DataFrame({"category": ["matched"], "phi": [np.pi - 0.01],
                           "theta": [np.pi / 2], "qop": [1.0],
                           "tpx": [-1.0], "tpy": [-0.01], "tpz": [0.0]})
    # True phi is just below -pi + 0.01: the residual is small, not about 2 pi
    assert abs(rp.add_residuals(tracks)["res_phi"].iloc[0]) < 0.05


def test_binned_efficiency():
    particles = pd.DataFrame({"pt": [0.5, 0.7, 1.5, 1.6], "reconstructable": True,
                              "found": [True, False, True, True]})
    x, eff, _ = rp.binned_efficiency(particles, "pt", np.array([0.0, 1.0, 2.0, 3.0]))
    assert list(x) == [0.5, 1.5, 2.5]
    assert eff[0] == pytest.approx(0.5) and eff[1] == pytest.approx(1.0)
    assert np.isnan(eff[2])                             # empty bin


# --- finding the runs and the whole program -----------------------------------

def test_find_runs(tmp_path):
    for run in ("run_10", "run_500"):
        write_event(tmp_path / "data" / run, tmp_path / "reco" / f"gpu_{run}_tracks")
    (tmp_path / "reco" / "gpu_run_7_tracks").mkdir()   # no truth: skipped
    runs = rp.find_runs([str(tmp_path / "reco")], tmp_path / "data")
    assert [(r[1], r[2]) for r in runs] == [("run_10", 10), ("run_500", 500)]


def run_script(*args):
    '''@brief runs the analysis as a program, without a display'''
    env = dict(os.environ, MPLBACKEND="Agg")
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True, env=env, check=False)


def test_program_single_run(tmp_path):
    write_event(tmp_path / "run_5", tmp_path / "reco_gpu")
    result = run_script("--tracks", tmp_path / "reco_gpu", "--truth", tmp_path / "run_5",
                        "-n", "single", "-o", tmp_path / "out")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "single.pdf").stat().st_size > 0
    table = pd.read_csv(tmp_path / "out" / "single.csv")
    # The table is written with 4 significant digits
    assert table["efficiency"].iloc[0] == pytest.approx(2 / 3, abs=1e-3)


def test_program_several_runs(tmp_path):
    for run in ("run_10", "run_500"):
        write_event(tmp_path / "data" / run, tmp_path / "reco" / f"gpu_{run}_tracks")
    result = run_script(tmp_path / "reco", "--data", tmp_path / "data", "-n", "scan",
                        "-o", tmp_path / "out")
    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(tmp_path / "out" / "scan.csv")) == 2


@pytest.mark.parametrize("args", [["--tracks", "x"], [], ["somewhere"]])
def test_program_rejects_incomplete_options(tmp_path, args):
    result = run_script(*args, "-o", tmp_path / "out")
    assert result.returncode != 0