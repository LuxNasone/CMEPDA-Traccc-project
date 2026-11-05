def parse_local_key(value):
    return int(value) if value.isdigit() else ord(value)

class Hit:

    def __init__(self, hit_id, row):
        self.hit_id = hit_id
        self.particle_id = int(row["particle_id"])
        self.geometry_id = int(row["geometry_id"])
        self.position = (float(row["tx"]), float(row["ty"]),
                                   float(row["tz"]), float(row["tt"]))
        self.momentum = (float(row["tpx"]), float(row["tpy"]), float(row["tpz"]), float(row["te"]))

    def __repr__(self):
        return f"Hit(id={self.hit_id}, particle={self.particle_id}, position={self.position})"

class Initials:

    def __init__(self, row):
        self.particle_id = int(row["particle_id"])
        self.particle_type = int(row["particle_type"])
        self.vertex = (float(row["vx"]), float(row["vy"]),
                       float(row["vz"]), float(row["vt"]))
        self.momentum = (float(row["px"]), float(row["py"]), float(row["pz"]))
        self.charge = float(row["q"])

    def __repr__(self):
        return (f"Initials(id={self.particle_id}, type={self.particle_type}, "f"momentum={self.momentum}, charge={self.charge})")

class Measurement:

    def __init__(self, row):
        self.measurement_id = int(row["measurement_id"])
        self.geometry_id = int(row["geometry_id"])
        self.local_key = parse_local_key(row["local_key"])
        self.local = (float(row["local0"]), float(row["local1"]))
        self.local_var = (float(row["var_local0"]), float(row["var_local1"]))
        self.direction = (float(row["phi"]), float(row["theta"]))
        self.time = float(row["time"])

    def __repr__(self):
        return (f"Measurement(id={self.measurement_id}," f"module={self.geometry_id}, local={self.local})")

class MeasurementHitMap:

    def __init__(self, row):
        self.measurement_id = int(row["measurement_id"])
        self.hit_id = int(row["hit_id"])

    def __repr__(self):
        return f"MeasurementHitMap(measurement={self.measurement_id}, hit={self.hit_id})"