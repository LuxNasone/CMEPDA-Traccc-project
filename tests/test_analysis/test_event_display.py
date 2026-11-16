'''
@brief Tests of src/analysis/event_display.py, on a small synthetic barrel
       detector written in a temporary folder.
'''

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import event_display as ed

## The script, to run it as a program
SCRIPT = Path(ed.__file__)

## Radii (mm) and number of modules in phi of the synthetic barrel layers
RADII = (30.0, 70.0, 120.0)
MODULES_PHI = (8, 16, 24)


def module_id(layer, phi_index, z_index):
    '''
    @brief geometry_id of a synthetic module: layer in the millions, then the
           module index in phi and in z (z_index from -49 to 49)
    '''
    return (layer + 1) * 1_000_000 + phi_index * 100 + (z_index + 50)


def write_events(folder, n_events=4, n_particles=6, seed=0):
    '''
    @brief writes the hits, particles and cells of straight tracks crossing
           every barrel layer, in the formats of the toy MC and the digitizer
    @param folder : (Path) run folder
    @return the folder
    '''
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for event in range(n_events):
        hits, particles, cells = [], [], []
        for pid in range(n_particles):
            phi = rng.uniform(-np.pi, np.pi)
            cot = rng.uniform(-1, 1)
            pt = rng.uniform(0.5, 5)
            particles.append({"particle_id": pid, "particle_type": 13, "process": 0,
                              "vx": 0, "vy": 0, "vz": 0, "vt": 0,
                              "px": pt * np.cos(phi), "py": pt * np.sin(phi),
                              "pz": pt * cot, "m": 0, "q": -1})
            for layer, (radius, n_phi) in enumerate(zip(RADII, MODULES_PHI)):
                # Flat modules: the radius changes a little across the module
                r = radius + rng.uniform(0, 2)
                z = r * cot
                gid = module_id(layer, int((phi + np.pi) / (2 * np.pi) * n_phi) % n_phi,
                                int(z // 50))
                hits.append({"particle_id": pid, "geometry_id": gid,
                             "tx": r * np.cos(phi), "ty": r * np.sin(phi), "tz": z})
                cells.append({"geometry_id": gid, "measurement_id": len(hits) - 1,
                              "channel0": rng.integers(0, 480),
                              "channel1": rng.integers(0, 1600),
                              "timestamp": 0, "value": 5000.0})
        cells.append({"geometry_id": module_id(0, 0, 0), "measurement_id": ed.NOISE_ID,
                      "channel0": 1, "channel1": 2, "timestamp": 0, "value": 1100.0})
        name = f"event{event:09d}"
        pd.DataFrame(hits).to_csv(folder / f"{name}-hits.csv", index=False)
        pd.DataFrame(particles).to_csv(folder / f"{name}-particles_initial.csv", index=False)
        pd.DataFrame(cells).to_csv(folder / f"{name}-cells.csv", index=False)
    return folder


def test_available_events(tmp_path):
    write_events(tmp_path, n_events=3)
    (tmp_path / "toy_digitization.json").write_text("{}")
    assert ed.available_events(tmp_path) == [0, 1, 2]


def test_read_many_adds_the_event_number(tmp_path):
    write_events(tmp_path, n_events=2, n_particles=2)
    hits = ed.read_many(tmp_path, [0, 1], "hits", ["particle_id", "tx"])
    assert list(hits.columns) == ["particle_id", "tx", "event"]
    assert sorted(hits["event"].unique()) == [0, 1]


def test_read_many_without_files_stops(tmp_path):
    with pytest.raises(SystemExit):
        ed.read_many(tmp_path, [0], "hits")


def test_add_cylindrical():
    hits = ed.add_cylindrical(pd.DataFrame({"tx": [3.0, 0.0], "ty": [4.0, -2.0]}))
    assert list(hits["r"]) == pytest.approx([5.0, 2.0])
    assert hits["phi"].iloc[1] == pytest.approx(-np.pi / 2)


def test_find_barrel_layers(tmp_path):
    write_events(tmp_path, n_events=20, n_particles=10)
    hits = ed.add_cylindrical(ed.read_many(tmp_path, range(20), "hits"))
    layers, radii = ed.find_barrel_layers(hits, min_hits=2)
    assert len(layers) == len(RADII)
    assert radii == pytest.approx(RADII, abs=2.0)
    # Every module of a layer belongs to that layer of the synthetic detector
    for index, modules in layers.items():
        assert {gid // 1_000_000 - 1 for gid in modules} == {index}


def test_find_barrel_layers_without_barrel():
    # Endcap-like modules: spread in r, not in z
    hits = pd.DataFrame({"geometry_id": [1] * 4, "tz": [500.0] * 4,
                         "r": [40.0, 60.0, 80.0, 100.0]})
    assert ed.find_barrel_layers(hits) == ({}, [])


def test_particle_pt():
    particles = pd.DataFrame({"particle_id": [7], "px": [3.0], "py": [4.0]})
    assert ed.particle_pt(particles)[7] == pytest.approx(5.0)


def test_program_writes_pdf_and_images(tmp_path):
    run = write_events(tmp_path / "run_10", n_events=40, n_particles=10)
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(run), "-o", str(tmp_path / "out"),
         "--png-dir", str(tmp_path / "png"), "--layers", "0", "-1"],
        capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "run_10_display.pdf").stat().st_size > 0
    images = {p.name for p in (tmp_path / "png").glob("*.png")}
    assert {"event_xy.png", "event_rz.png", "hits_xy.png", "hits_rz.png",
            "layer0_phi_z.png", "layer2_occupancy.png"} <= images


def test_program_rejects_a_missing_event(tmp_path):
    run = write_events(tmp_path / "run_10", n_events=2)
    result = subprocess.run([sys.executable, str(SCRIPT), str(run), "--event", "99"],
                            capture_output=True, text=True,
                            env=dict(os.environ, MPLBACKEND="Agg"), check=False)
    assert result.returncode != 0
    assert "event 99 not found" in result.stderr