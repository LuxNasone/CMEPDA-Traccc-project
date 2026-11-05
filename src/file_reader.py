import numpy as np

def read_hit(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1, ndmin = 2)
    hits = {}

    for row in data:
        particle_id = int(row[0])
        four_point = tuple(row[2:6])     # tx, ty, tz, tt
        four_momentum = tuple(row[6:10]) # tpx, tpy, tpz, te

        hits.setdefault(particle_id, []).append((four_point, four_momentum))

    return hits