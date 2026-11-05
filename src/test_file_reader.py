from file_reader import *

HEADER_HITS = "particle_id,geometry_id,tx,ty,tz,tt,tpx,tpy,tpz,te,deltapx,deltapy,deltapz,deltae,index\n"
HEADER_INITIALS = "particle_id,particle_type,process,vx,vy,vz,vt,px,py,pz,m,q\n"
HEADER_MEAS = "measurement_id,geometry_id,local_key,local0,local1,phi,theta,time,var_local0,var_local1,var_phi,var_theta,var_time\n"
HEADER_MAP = "measurement_id,hit_id\n"


def write_csv(tmp_path, header, rows):
    path = tmp_path / "data.csv"
    path.write_text(header + rows)
    return path


# Tests for read_hits

def test_every_row_is_a_hit(tmp_path):
    path = write_csv(tmp_path, HEADER_HITS,
        "0,100,1,2,3,4,5,6,7,0,0,0,0,0,0\n"
        "0,101,8,9,10,11,12,13,14,0,0,0,0,0,0\n"
        "1,100,15,16,17,18,19,20,21,0,0,0,0,0,0\n")
    hits = read_hits(path)
    assert set(hits) == {0, 1, 2}
    assert [hit.particle_id for hit in hits.values()] == [0, 0, 1]


def test_hit_right_columns(tmp_path):
    path = write_csv(tmp_path, HEADER_HITS, "3,100,1,2,4,5,6,7,8,9,0,0,0,0,0\n")
    hit = read_hits(path)[0]
    assert hit.hit_id == 0
    assert hit.particle_id == 3
    assert hit.geometry_id == 100
    assert hit.position == (1, 2, 4, 5)
    assert hit.momentum == (6, 7, 8, 9)


def test_hit_id_is_row_number(tmp_path):
    path = write_csv(tmp_path, HEADER_HITS,
        "0,100,1,2,3,4,5,6,7,0,0,0,0,0,0\n"
        "1,100,8,9,10,11,12,13,14,0,0,0,0,0,0\n"
        "0,101,15,16,17,18,19,20,21,0,0,0,0,0,0\n")
    hits = read_hits(path)
    assert hits[2].particle_id == 0
    assert hits[2].geometry_id == 101


# Tests for read_initials

def test_initials_right_columns(tmp_path):
    path = write_csv(tmp_path, HEADER_INITIALS, "7,13,99,1,2,3,4,5,6,8,9,-1\n")
    particle = read_initials(path)[7]
    assert particle.particle_id == 7
    assert particle.particle_type == 13
    assert particle.vertex == (1, 2, 3, 4)
    assert particle.momentum == (5, 6, 8)
    assert particle.charge == -1


def test_initials_more_particles(tmp_path):
    path = write_csv(tmp_path, HEADER_INITIALS,
        "0,13,0,0,0,0,0,1,2,3,0,-1\n"
        "1,-13,0,0,0,0,0,4,5,6,0,1\n")
    particles = read_initials(path)
    assert set(particles) == {0, 1}
    assert particles[1].particle_type == -13
    assert particles[1].charge == 1


# Tests for read_measurements

def test_measurement_right_columns(tmp_path):
    path = write_csv(tmp_path, HEADER_MEAS, "5,100,6,1,2,3,4,7,8,9,0,0,0\n")
    meas = read_measurements(path)[5]
    assert meas.measurement_id == 5
    assert meas.geometry_id == 100
    assert meas.local_key == 6
    assert meas.local == (1, 2)
    assert meas.local_var == (8, 9)
    assert meas.direction == (3, 4)
    assert meas.time == 7

#Test for read_map

def test_map_connects_measurement_to_hit(tmp_path):
    path = write_csv(tmp_path, HEADER_MAP, "0,4\n1,7\n")
    meas_map = read_map(path)
    assert set(meas_map) == {0, 1}
    assert meas_map[0].hit_id == 4
    assert meas_map[1].hit_id == 7

#Test combined test

def test_measurement_to_particle(tmp_path):
    hits = read_hits(write_csv(tmp_path, HEADER_HITS,
        "3,100,1,2,4,5,6,7,8,9,0,0,0,0,0\n"))
    particles = read_initials(write_csv(tmp_path, HEADER_INITIALS,
        "3,13,99,1,2,3,4,5,6,8,9,-1\n"))
    meas_map = read_map(write_csv(tmp_path, HEADER_MAP, "0,0\n"))

    hit = hits[meas_map[0].hit_id]
    assert particles[hit.particle_id].particle_type == 13

#Test for file clustering

def test_group_event_files(tmp_path):
    for name in ["event000000000-hits.csv", "event000000000-measurements.csv",
                 "event000000001-hits.csv", "note.txt"]:
        (tmp_path / name).touch()

    events = group_event_files(tmp_path)
    assert set(events) == {0, 1}
    assert set(events[0]) == {"hits", "measurements"}
    assert events[1]["hits"].name == "event000000001-hits.csv"