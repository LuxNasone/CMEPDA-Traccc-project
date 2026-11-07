import json
from pathlib import Path

import numpy as np
import pandas as pd

## measurement_id written for the pixels fired by noise alone
NOISE_ID = np.iinfo(np.uint64).max

## Columns of the -cells.csv file, in the order read by traccc
CELL_COLUMNS = ["geometry_id", "measurement_id", "channel0", "channel1",
                "timestamp", "value"]

## Columns of -measurements.csv used by the digitization
MEASUREMENT_COLUMNS = ["measurement_id", "geometry_id", "local0", "local1", "time"]


class EventIO:
    '''
    @brief Reads the measurements of the toy MC, digitizes them and writes the
           files read by traccc
    '''

    def __init__(self, dgtz):
        '''
        @brief Default constructor
        @param dgtz : (Digitizer) digitizer applied to every event
        '''
        self.dgtz = dgtz

    @staticmethod
    def read_measurements(path):
        '''
        @brief Reads the columns of a -measurements.csv file used by the digitization
        @param path : (str or Path) measurement file
        @return (pandas.DataFrame) measurement_id, geometry_id, local0, local1, time.
                The IDs are unsigned 64 bit integers, so they keep their precision.
        '''
        return pd.read_csv(
            path,
            usecols=MEASUREMENT_COLUMNS,
            dtype={"measurement_id": np.uint64, "geometry_id": np.uint64,
                   "local0": float, "local1": float, "time": float},
        )

    @staticmethod
    def collect_modules(paths):
        '''
        @brief Modules hit at least once in the run: noisy pixels are generated on them
        @param paths : (iterable of str or Path) all the measurement files of the run
        @return (np.array of uint64) sorted module IDs
        '''
        modules = set()
        for path in paths:
            ids = pd.read_csv(path, usecols=["geometry_id"], dtype={"geometry_id": np.uint64})
            modules.update(ids["geometry_id"].tolist())
        return np.array(sorted(modules), dtype=np.uint64)

    @staticmethod
    def cells_table(meas, source, module, channel_0, channel_1, value):
        '''
        @brief Builds the table of cells from the output of Digitizer.digitize
        @param meas : (pandas.DataFrame) measurements of the event
        @param source, module, channel_0, channel_1, value : output of digitize.
               source is the row of the measurement that lit the pixel, -1 for noise.
        @return (pandas.DataFrame) one row per cell, with the CELL_COLUMNS
        '''
        source = np.asarray(source)
        signal = source >= 0  # -1 would silently pick the last measurement

        measurement_id = np.full(len(source), NOISE_ID, dtype=np.uint64)
        timestamp = np.zeros(len(source))
        measurement_id[signal] = meas["measurement_id"].to_numpy(np.uint64)[source[signal]]
        timestamp[signal] = meas["time"].to_numpy(float)[source[signal]]

        return pd.DataFrame({
            "geometry_id": np.asarray(module, dtype=np.uint64),
            "measurement_id": measurement_id,
            "channel0": np.asarray(channel_0, dtype=np.int64),
            "channel1": np.asarray(channel_1, dtype=np.int64),
            "timestamp": timestamp,
            "value": np.asarray(value, dtype=float),
        })[CELL_COLUMNS]

    def start_digitization(self, path, modules, rng):
        '''
        @brief Digitizes the measurements of one event
        @param path : (str or Path) measurement file of the event
        @param modules : (np.array of uint64) modules of the run, from collect_modules
        @param rng : (numpy.random.Generator) generator of the event
        @return (pandas.DataFrame) cells of the event, sorted by module and channels
        '''
        meas = self.read_measurements(path)
        output = self.dgtz.digitize(meas["local0"].to_numpy(), meas["local1"].to_numpy(),
                                    meas["geometry_id"].to_numpy(np.uint64), modules, rng)
        cells = self.cells_table(meas, *output)
        return (cells.sort_values(["geometry_id", "channel1", "channel0"], kind="stable")
                     .reset_index(drop=True))

    @staticmethod
    def cells_path(folder, event):
        '''
        @brief Path of the cell file of an event, e.g. event000000003-cells.csv
        @param folder : (str or Path) output folder
        @param event : (int) event number
        @return (Path) path of the file
        '''
        return Path(folder) / f"event{event:09d}-cells.csv"

    @staticmethod
    def write_cells(path, cells):
        '''
        @brief Writes the -cells.csv file read by traccc
        @param path : (str or Path) output file
        @param cells : (pandas.DataFrame) cells, with the CELL_COLUMNS
        '''
        cells[CELL_COLUMNS].to_csv(path, index=False, float_format="%.6g")

    @staticmethod
    def write_digitization(path, grid):
        '''
        @brief Writes the digitization file read by traccc
        @param path : (str or Path) output file
        @param grid : (PixelGrid) pixel matrix, the same for all modules

        There is a single entry, without volume, layer and sensitive: in ACTS an
        identifier made of zeros is the default for every module. It is needed
        because the toy detector of detray gives all modules the same ACTS ID.
        The same file is passed to traccc as --conditions-file too.
        '''
        binning = [
            {"type": "equidistant", "bins": n, "min": minimum, "max": minimum + n * grid.pitch}
            for n, minimum in ((grid.n_0, grid.min_0), (grid.n_1, grid.min_1))
        ]
        content = {"entries": [
            {"value": {"geometric": {"segmentation": {"binningdata": binning}}}}
        ]}
        with open(path, "w") as f:
            json.dump(content, f, indent=2)
    