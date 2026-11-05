import numpy as np

def read_hit(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1, ndmin = 2)
    hits = {}

    for row in data:
        particle_id = int(row[0])
        four_point = tuple(row[2:6])     
        four_momentum = tuple(row[6:10]) 

        hits.setdefault(particle_id, []).append((four_point, four_momentum))

    return hits

def read_initials(filename):
    data = np.genfromtxt(filename, delimiter=",", skip_header=1, ndmin = 2)
    initials = {}

    for row in data:
        particle_id = int(row[0])
        particle_type = int(row[1])
        four_vertex = tuple(row[3:7])
        px_py_pz_m = tuple(row[7:11])
        charge = int(row[11])

        initials[particle_id]= (particle_type, four_vertex, px_py_pz_m, charge)

    return initials