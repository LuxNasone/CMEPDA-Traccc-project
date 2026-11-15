'''
@brief Creates the cells read by traccc from the measurements of the toy MC.

The events are independent (each has its own random generator, see event_rng),
so they are digitized in parallel by a pool of processes, one event per task.
The cells do not depend on the number of processes.

Usage:
    python src/digitize/toy_digitize.py <event folder> [--config config/digitize.conf]
                                                       [--workers N]
'''

import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from charge import ChargeCloud, LandauDeposit
from digitizer import Digitizer
from event_io import NOISE_ID, EventIO
from pixel_grid import PixelGrid
from readout import Readout

logger = logging.getLogger(__name__)

## Name of the measurement files written by the toy MC
MEASUREMENT_FILE = re.compile(r"event(\d+)-measurements\.csv")

## Name of the digitization file written next to the cells
DIGITIZATION_FILE = "toy_digitization.json"

UM = 1e-3  # 1 um in mm

## Keys of the configuration file: key -> (type, default)
CONFIG_KEYS = {
    "pitch-um": (float, 50.0),
    "half-size0-mm": (float, 12.0),
    "half-size1-mm": (float, 40.0),
    "landau-mpv-e-per-um": (float, 76.0),
    "landau-width-e-per-um": (float, 5.0),
    "thickness-um": (float, 110.0),
    "diffusion-sigma-um": (float, 10.0),
    "diffusion-n-sigma": (float, 4.0),
    "noise-mean-e": (float, 0.0),
    "noise-sigma-e": (float, 100.0),
    "threshold-e": (float, 1000.0),
    "noisy-pixel-fraction": (float, 1e-6),
    "seed": (int, 0),
}


def read_config(path):
    '''
    @brief Reads the configuration file
    @param path : (str or Path) file with one "key = value" per line, "#" for comments
    @return (dict) value of every key in CONFIG_KEYS; missing keys get their default
    @throws ValueError if a line, a key or a value is not valid
    '''
    config = {key: default for key, (_, default) in CONFIG_KEYS.items()}
    with open(path) as f:
        for number, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{number}: invalid line: {raw.strip()}")

            key, value = (part.strip() for part in line.split("=", 1))
            if key not in CONFIG_KEYS:
                raise ValueError(f"{path}:{number}: unknown key: {key}")
            kind, _ = CONFIG_KEYS[key]
            try:
                config[key] = kind(value)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: invalid value for {key}: {value}") \
                    from error
    return config


def build_digitizer(config):
    '''
    @brief Creates the Digitizer described by the configuration
    @param config : (dict) output of read_config
    @return (Digitizer) digitizer with lengths in mm and charges in electrons
    '''
    grid = PixelGrid(config["pitch-um"] * UM, config["half-size0-mm"],
                     config["half-size1-mm"])
    deposit = LandauDeposit(config["landau-mpv-e-per-um"], config["landau-width-e-per-um"],
                            config["thickness-um"], config["seed"])
    cloud = ChargeCloud(config["diffusion-sigma-um"] * UM, config["diffusion-n-sigma"])
    readout = Readout(config["noise-mean-e"], config["noise-sigma-e"], config["threshold-e"],
                      config["noisy-pixel-fraction"], config["seed"])
    return Digitizer(grid, deposit, cloud, readout)


def find_measurement_files(folder):
    '''
    @brief Measurement files of a folder, by event number
    @param folder : (str or Path) folder written by the toy MC
    @return (dict) {event number: path}, sorted by event
    '''
    found = {}
    for path in Path(folder).iterdir():
        match = MEASUREMENT_FILE.fullmatch(path.name)
        if match is not None:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


def event_rng(seed, event):
    '''
    @brief Generator of one event: it depends only on the seed and the event number,
           so every event is reproducible and independent of the others
    '''
    return np.random.default_rng([seed, event])


## State of a worker process, set once by init_worker and reused for all its events
_worker = {}


def init_worker(config, modules, output):
    '''
    @brief Prepares a worker process: builds its own Digitizer once, so that it
           is not sent again with every event
    @param config : (dict) output of read_config
    @param modules : modules with noisy pixels, from EventIO.collect_modules
    @param output : (Path) folder where the cells are written
    '''
    _worker["io"] = EventIO(build_digitizer(config))
    _worker["modules"] = modules
    _worker["output"] = output
    _worker["seed"] = config["seed"]


def digitize_event(task):
    '''
    @brief Digitizes one event and writes its cells (runs in a worker process)
    @param task : (tuple) (event number, path of its -measurements.csv file)
    @return (tuple) (event number, number of cells, number of cells from noise)
    '''
    event, path = task
    io = _worker["io"]
    cells = io.start_digitization(path, _worker["modules"],
                                  event_rng(_worker["seed"], event))
    EventIO.write_cells(EventIO.cells_path(_worker["output"], event), cells)
    n_noise = int((cells["measurement_id"] == NOISE_ID).sum())
    return event, len(cells), n_noise


def digitize_folder(folder, output, config, workers=1):
    '''
    @brief Digitizes all the events of a folder
    @param folder : (Path) folder with the -measurements.csv files
    @param output : (Path) folder where the cells and the digitization file are written
    @param config : (dict) output of read_config
    @param workers : (int) number of processes; 1 = no pool, everything in this process
    @return (int) number of events digitized
    '''
    events = find_measurement_files(folder)
    if not events:
        raise FileNotFoundError(f"no -measurements.csv file in {folder}")
    output.mkdir(parents=True, exist_ok=True)

    io = EventIO(build_digitizer(config))
    grid = io.dgtz.grid
    logger.info("Configuration: %s", config)
    logger.info("Pixel matrix: %d x %d pixels", grid.n_0, grid.n_1)

    modules = EventIO.collect_modules(events.values())
    logger.info("Modules with noisy pixels: %d", len(modules))

    tasks = list(events.items())
    workers = max(1, min(workers, len(tasks)))
    logger.info("Digitizing %d events with %d process(es)", len(tasks), workers)

    if workers == 1:
        init_worker(config, modules, output)
        results = map(digitize_event, tasks)
        for event, n_cells, n_noise in results:
            logger.info("Event %d: %d cells (%d from noise)", event, n_cells, n_noise)
    else:
        # Several events per task when they are many, to limit the overhead
        chunksize = max(1, len(tasks) // (4 * workers))
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=(config, modules, output)) as pool:
            # map returns the results in event order; the log is written here,
            # in the main process, so the workers never write the same file
            for event, n_cells, n_noise in pool.map(digitize_event, tasks,
                                                    chunksize=chunksize):
                logger.info("Event %d: %d cells (%d from noise)", event, n_cells, n_noise)

    EventIO.write_digitization(output / DIGITIZATION_FILE, grid)
    return len(events)


def main():
    parser = argparse.ArgumentParser(description="Digitizes the measurements of the toy MC.")
    parser.add_argument("folder", type=Path, help="folder with the -measurements.csv files")
    parser.add_argument("--config", type=Path, default=Path("config/digitize.conf"),
                        help="configuration file")
    parser.add_argument("--output", type=Path, default=None,
                        help="folder of the cells (default: the event folder)")
    parser.add_argument("--log-file", type=Path, default=Path("logs/toy_digitize.log"),
                        help="log file")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="number of processes digitizing events in parallel "
                             "(default: number of cores; 1 = no parallelism)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log_file, filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.folder.is_dir():
        parser.error(f"folder not found: {args.folder}")
    try:
        config = read_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output = args.output or args.folder
    try:
        n_events = digitize_folder(args.folder, output, config, args.workers)
    except FileNotFoundError as error:
        parser.error(str(error))

    print(f"Digitized {n_events} events in {output}. "
          f"Digitization file: {output / DIGITIZATION_FILE}. Log in {args.log_file}")


if __name__ == "__main__":
    main()