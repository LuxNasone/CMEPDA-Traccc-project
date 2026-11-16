import json

import numpy as np
import pandas as pd
import pytest

from charge import ChargeCloud, LandauDeposit
from digitizer import Digitizer
from event_io import CELL_COLUMNS, NOISE_ID, EventIO
from pixel_grid import PixelGrid
from readout import Readout

HEADER = ("measurement_id,geometry_id,local_key,local0,local1,phi,theta,"
          "time,var_local0,var_local1,var_phi,var_theta,var_time\n")
BIG_ID = 2**63 + 5
MOST_PROBABLE = 76.0 * 100.0  # electrons, for a 100 um sensor


def make_io(fraction=0.0, thr=0.0):
    '''EventIO with a digitizer without randomness (unless noise is asked).'''
    digitizer = Digitizer(PixelGrid(0.05, 1.0, 2.0), LandauDeposit(76.0, 0.0, 100.0, seed=0),
                          ChargeCloud(0.0), Readout(0.0, 0.0, thr, fraction, seed=0))
    return EventIO(digitizer)


def write_measurements(path, rows):
    '''rows: list of (measurement_id, geometry_id, local0, local1, time)'''
    lines = [f"{m},{g},\x06,{l0},{l1},0.1,0.2,{t},0,0,0,0,0\n" for m, g, l0, l1, t in rows]
    path.write_text(HEADER + "".join(lines))
    return path


# read_measurements

def test_read_measurements_keeps_large_ids(tmp_path):
    path = write_measurements(tmp_path / "event000000000-measurements.csv",
                              [(3, BIG_ID, 1.5, -2.5, 7.0)])
    meas = EventIO.read_measurements(path)
    assert list(meas.columns) == ["measurement_id", "geometry_id", "local0", "local1", "time"]
    assert meas["geometry_id"].iloc[0] == BIG_ID
    assert meas["measurement_id"].iloc[0] == 3
    assert (meas["local0"].iloc[0], meas["local1"].iloc[0]) == (1.5, -2.5)
    assert meas["time"].iloc[0] == 7.0


# collect_modules

def test_collect_modules_joins_all_events(tmp_path):
    a = write_measurements(tmp_path / "a.csv", [(0, 8, 0, 0, 0), (1, 7, 0, 0, 0)])
    b = write_measurements(tmp_path / "b.csv", [(0, 7, 0, 0, 0), (1, BIG_ID, 0, 0, 0)])
    modules = EventIO.collect_modules([a, b])
    assert modules.dtype == np.uint64
    assert modules.tolist() == [7, 8, BIG_ID]


# cells_table

def test_cells_table_signal_and_noise():
    meas = pd.DataFrame({"measurement_id": np.array([10, 11], dtype=np.uint64),
                         "geometry_id": np.array([7, 7], dtype=np.uint64),
                         "local0": [0.0, 0.0], "local1": [0.0, 0.0], "time": [1.0, 2.0]})
    cells = EventIO.cells_table(meas, source=[1, -1], module=[7, 7], channel_0=[3, 4],
                                channel_1=[5, 6], value=[100.0, 200.0])
    assert list(cells.columns) == CELL_COLUMNS
    assert cells["measurement_id"].tolist() == [11, NOISE_ID]
    assert cells["timestamp"].tolist() == [2.0, 0.0]


def test_noise_does_not_take_last_measurement():
    meas = pd.DataFrame({"measurement_id": np.array([10], dtype=np.uint64),
                         "geometry_id": np.array([7], dtype=np.uint64),
                         "local0": [0.0], "local1": [0.0], "time": [5.0]})
    cells = EventIO.cells_table(meas, [-1], [7], [3], [5], [100.0])
    assert cells["measurement_id"].iloc[0] == NOISE_ID
    assert cells["timestamp"].iloc[0] == 0.0


# start_digitization

def test_start_digitization_one_hit(tmp_path):
    path = write_measurements(tmp_path / "m.csv", [(4, BIG_ID, 0.012, 0.3, 0.0)])
    io = make_io()
    cells = io.start_digitization(path, np.array([BIG_ID], dtype=np.uint64),
                                  np.random.default_rng(0))
    col, row = io.dgtz.grid.convert_to_channel(0.012, 0.3)
    assert len(cells) == 1
    cell = cells.iloc[0]
    assert cell["geometry_id"] == BIG_ID
    assert cell["measurement_id"] == 4
    assert (cell["channel0"], cell["channel1"]) == (col, row)
    assert cell["value"] == pytest.approx(MOST_PROBABLE)


def test_start_digitization_sorts_cells(tmp_path):
    path = write_measurements(tmp_path / "m.csv", [(0, 8, 0.5, 1.0, 0), (1, 7, -0.5, 1.0, 0),
                                                   (2, 7, 0.5, -1.0, 0)])
    cells = make_io().start_digitization(path, np.array([7, 8], dtype=np.uint64),
                                         np.random.default_rng(0))
    keys = list(zip(cells["geometry_id"], cells["channel1"], cells["channel0"]))
    assert keys == sorted(keys)


def test_start_digitization_with_noise(tmp_path):
    path = write_measurements(tmp_path / "m.csv", [(0, 7, 0.012, 0.3, 0)])
    cells = make_io(fraction=1e-2, thr=1000.0).start_digitization(
        path, np.array([7, 8], dtype=np.uint64), np.random.default_rng(0))
    assert 0 in cells["measurement_id"].tolist()
    assert NOISE_ID in cells["measurement_id"].tolist()


# cells_path and write_cells

def test_cells_path():
    assert EventIO.cells_path("data/toy", 3).as_posix() == "data/toy/event000000003-cells.csv"


def test_write_cells_format(tmp_path):
    cells = pd.DataFrame({"geometry_id": np.array([BIG_ID], dtype=np.uint64),
                          "measurement_id": np.array([NOISE_ID], dtype=np.uint64),
                          "channel0": [3], "channel1": [4],
                          "timestamp": [0.0], "value": [1234.5]})
    path = EventIO.cells_path(tmp_path, 2)
    EventIO.write_cells(path, cells)
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(CELL_COLUMNS)
    assert lines[1] == f"{BIG_ID},{NOISE_ID},3,4,0,1234.5"


def test_written_cells_can_be_read_back(tmp_path):
    path = write_measurements(tmp_path / "m.csv", [(0, BIG_ID, 0.012, 0.3, 0)])
    io = make_io()
    cells = io.start_digitization(path, np.array([BIG_ID], dtype=np.uint64),
                                  np.random.default_rng(0))
    out = EventIO.cells_path(tmp_path, 0)
    EventIO.write_cells(out, cells)
    back = pd.read_csv(out, dtype={"geometry_id": np.uint64, "measurement_id": np.uint64})
    assert back["geometry_id"].tolist() == cells["geometry_id"].tolist()
    assert back["channel0"].tolist() == cells["channel0"].tolist()


# write_digitization

def test_write_digitization(tmp_path):
    path = tmp_path / "digi.json"
    EventIO.write_digitization(path, PixelGrid(0.05, 12.0, 40.0))
    entries = json.loads(path.read_text())["entries"]
    assert len(entries) == 1
    assert not {"volume", "layer", "sensitive"} & set(entries[0])
    binning = entries[0]["value"]["geometric"]["segmentation"]["binningdata"]
    assert [b["bins"] for b in binning] == [480, 1600]
    assert binning[0]["min"] == pytest.approx(-12.0)
    assert binning[0]["max"] == pytest.approx(12.0)
    assert binning[1]["min"] == pytest.approx(-40.0)
    assert binning[1]["max"] == pytest.approx(40.0)