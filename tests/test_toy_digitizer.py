import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from event_io import CELL_COLUMNS
from toy_digitize import (DIGITIZATION_FILE, build_digitizer, digitize_folder,
                          event_rng, find_measurement_files, read_config)

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "digitization" / "toy_digitize.py"
HEADER = ("measurement_id,geometry_id,local_key,local0,local1,phi,theta,"
          "time,var_local0,var_local1,var_phi,var_theta,var_time\n")
BIG_ID = 2**63 + 5


def write_event(folder, event, rows):
    '''rows: list of (geometry_id, local0, local1)'''
    lines = [f"{i},{g},\x06,{l0},{l1},0,0,0,0,0,0,0,0\n" for i, (g, l0, l1) in enumerate(rows)]
    path = folder / f"event{event:09d}-measurements.csv"
    path.write_text(HEADER + "".join(lines))
    return path


def write_config(path, text=""):
    path.write_text(text)
    return path


# read_config

def test_defaults_when_file_is_empty(tmp_path):
    config = read_config(write_config(tmp_path / "c.conf"))
    assert config["pitch-um"] == 50.0
    assert config["seed"] == 0


def test_values_and_comments(tmp_path):
    config = read_config(write_config(tmp_path / "c.conf",
                                      "# comment\npitch-um = 25\nseed = 3  # end of line\n"))
    assert config["pitch-um"] == 25.0
    assert config["seed"] == 3
    assert isinstance(config["seed"], int)


@pytest.mark.parametrize("text", ["pitch-um\n", "colour = blue\n", "pitch-um = x\n",
                                  "seed = 1.5\n"])
def test_invalid_lines(tmp_path, text):
    with pytest.raises(ValueError):
        read_config(write_config(tmp_path / "c.conf", text))


def digitization_section(path):
    '''
    Lines of the [digitization] section of config/simulation.conf, plus the seed
    of the [common] section: the file that run_simulation.sh gives to toy_digitize.
    '''
    section, lines = None, []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif line and (section == "digitization"
                       or (section == "common" and line.split("=")[0].strip() == "seed")):
            lines.append(line)
    return "\n".join(lines) + "\n"


def test_project_config_is_valid(tmp_path):
    path = Path(__file__).resolve().parents[1] / "config" / "simulation.conf"
    text = digitization_section(path)
    assert "pitch-um" in text  # the [digitization] section was found
    build_digitizer(read_config(write_config(tmp_path / "c.conf", text)))


# build_digitizer

def test_units_are_converted(tmp_path):
    config = read_config(write_config(tmp_path / "c.conf",
                                      "pitch-um = 50\ndiffusion-sigma-um = 10\n"))
    digitizer = build_digitizer(config)
    assert digitizer.grid.pitch == pytest.approx(0.05)
    assert digitizer.cloud.sigma == pytest.approx(0.01)
    assert digitizer.grid.n_0 == 480


# find_measurement_files and event_rng

def test_find_measurement_files(tmp_path):
    for name in ["event000000001-measurements.csv", "event000000000-measurements.csv",
                 "event000000000-hits.csv", "event000000000-cells.csv", "notes.txt"]:
        (tmp_path / name).touch()
    assert list(find_measurement_files(tmp_path)) == [0, 1]


def test_event_rng_depends_on_seed_and_event():
    assert event_rng(0, 1).random() == event_rng(0, 1).random()
    assert event_rng(0, 1).random() != event_rng(0, 2).random()
    assert event_rng(0, 1).random() != event_rng(1, 1).random()


# digitize_folder

def test_digitize_folder_writes_cells_and_json(tmp_path):
    write_event(tmp_path, 0, [(BIG_ID, 0.012, 0.3), (7, 1.0, -5.0)])
    write_event(tmp_path, 1, [(7, -2.0, 10.0)])
    config = read_config(write_config(tmp_path / "c.conf", "noisy-pixel-fraction = 0\n"))

    assert digitize_folder(tmp_path, tmp_path / "out", config) == 2

    for event in (0, 1):
        cells = pd.read_csv(tmp_path / "out" / f"event{event:09d}-cells.csv",
                            dtype={"geometry_id": np.uint64})
        assert list(cells.columns) == CELL_COLUMNS
        assert len(cells) > 0
    cells0 = pd.read_csv(tmp_path / "out" / "event000000000-cells.csv",
                         dtype={"geometry_id": np.uint64})
    assert BIG_ID in cells0["geometry_id"].tolist()

    digi = json.loads((tmp_path / "out" / DIGITIZATION_FILE).read_text())
    assert len(digi["entries"]) == 1


def test_digitize_folder_is_reproducible(tmp_path):
    write_event(tmp_path, 0, [(7, 0.0249, 0.3)] * 3)
    config = read_config(write_config(tmp_path / "c.conf", "noisy-pixel-fraction = 1e-4\n"))
    digitize_folder(tmp_path, tmp_path / "a", config)
    digitize_folder(tmp_path, tmp_path / "b", config)
    a = (tmp_path / "a" / "event000000000-cells.csv").read_text()
    b = (tmp_path / "b" / "event000000000-cells.csv").read_text()
    assert a == b


def test_empty_folder_is_an_error(tmp_path):
    config = read_config(write_config(tmp_path / "c.conf"))
    with pytest.raises(FileNotFoundError):
        digitize_folder(tmp_path, tmp_path / "out", config)


# Command line

def test_command_line(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    write_event(events, 0, [(7, 0.012, 0.3)])
    config = write_config(tmp_path / "c.conf", "seed = 2\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(events), "--config", str(config),
         "--log-file", str(tmp_path / "logs" / "digi.log")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (events / "event000000000-cells.csv").exists()
    assert (events / DIGITIZATION_FILE).exists()
    assert "Event 0" in (tmp_path / "logs" / "digi.log").read_text()


def test_command_line_missing_folder(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path / "missing"),
                             "--log-file", str(tmp_path / "l.log")],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "folder not found" in result.stderr