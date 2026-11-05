import numpy as np
import csv
from classes import Hit, Initials, Measurement, MeasurementHitMap
import re
from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__name__)

EVENT_FILE = re.compile(r"event(\d+)-(.+)\.csv")


def group_event_files(folder):
    events = {}
    for path in sorted(Path(folder).iterdir()):
        match = EVENT_FILE.fullmatch(path.name)
        if match is None:
            continue
        event_id = int(match.group(1))
        kind = match.group(2)
        events.setdefault(event_id, {})[kind] = path
    return events


def load_event(paths):
    return {
        "hits": read_hits(paths["hits"]),
        "particles": read_initials(paths["particles_initial"]),
        "measurements": read_measurements(paths["measurements"]),
        "map": read_map(paths["measurement-simhit-map"]),
    }


def read_hits(filename):

    with open(filename, newline="") as f:
        return {i: Hit(i, row) for i, row in enumerate(csv.DictReader(f))}

def read_objects(filename, cls, key):
    with open(filename, newline="") as f:
        objects = map(cls, csv.DictReader(f))
        return {getattr(obj, key): obj for obj in objects}


def read_initials(filename):
    return read_objects(filename, Initials, "particle_id")


def read_measurements(filename):
    return read_objects(filename, Measurement, "measurement_id")


def read_map(filename):
    return read_objects(filename, MeasurementHitMap, "measurement_id")

def main():
    parser = argparse.ArgumentParser(description="Analizza gli eventi del toy MC.")
    parser.add_argument("folder", type=Path, help="cartella con i CSV degli eventi")
    parser.add_argument("--log-file", type=Path, default=Path("logs/events.log"),help="file di log")
    args = parser.parse_args()

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log_file,filemode="w",level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")

    if not args.folder.is_dir():
        parser.error(f"cartella non trovata: {args.folder}")

    events = group_event_files(args.folder)
    if not events:
        parser.error(f"nessun file evento in: {args.folder}")

    for event_id, paths in sorted(events.items()):
        event = load_event(paths)
        logger.info("Evento %d: %d particelle, %d hit, %d misure", event_id, len(event["particles"]), len(event["hits"]), len(event["measurements"]))

    print(f"Letti {len(events)} eventi. Log in {args.log_file}")

if __name__ == "__main__":
    main()


