from file_reader import*
import pytest

HEADER_HITS = "particle_id,geometry_id,tx,ty,tz,tt,tpx,tpy,tpz,te,deltapx,deltapy,deltapz,deltae,index\n"
HEADER_INITIALS = "particle_id,particle_type,process,vx,vy,vz,vt,px,py,pz,m,q\n"

def write_csv(tmp_path, header, righe):
    path = tmp_path / "hits.csv"
    path.write_text(header + righe)
    return path

#Tests for read_hits

def test_cluster_by_particle(tmp_path):
    path = write_csv(tmp_path, HEADER_HITS,
        "0,100,1,2,3,4,5,6,7,0,0,0,0,0,0\n"
        "0,101,8,9,10,11,12,13,14,0,0,0,0,0,0\n"
        "1,100,15,16,17,18,19,20,21,0,0,0,0,0,0\n")
    hits = read_hit(path)
    assert set(hits) == {0, 1}
    assert len(hits[0]) == 2
    assert len(hits[1]) == 1

def test_right_column(tmp_path):
    path = write_csv(tmp_path, HEADER_HITS, "0,100,1,2,3,4,5,6,7,0,0,0,0,0,0\n")
    four_point, four_impulse = read_hit(path)[0][0]
    assert four_point == (1, 2, 3, 4)
    assert four_impulse == (5, 6, 7, 0)

def test_not_consecutive_hits(tmp_path):
    path = write_csv(tmp_path, HEADER_HITS,
        "0,100,1,2,3,4,5,6,7,0,0,0,0,0,0\n"
        "1,100,8,9,10,11,12,13,14,0,0,0,0,0,0\n"
        "0,101,15,16,17,18,19,20,21,0,0,0,0,0,0\n")
    assert len(read_hit(path)[0]) == 2

#Tests for read_initials

def test_right_column(tmp_path):
    path = write_csv(tmp_path, HEADER_INITIALS, "0,13,0,0,0,0,0,3.75868,0.331486,2.88878,0,-1\n")
    result = read_initials(path)[0]
    particle_type = result[0]
    four_vertex = result[1]
    px_py_pz_m = result[2]
    charge = result[3]
    assert particle_type == 13
    assert four_vertex == (0, 0, 0, 0)
    assert px_py_pz_m == pytest.approx((3.75868, 0.331486, 2.88878, 0))
    assert charge == -1


def test_different_values_in_columns(tmp_path):
    path = write_csv(tmp_path, HEADER_INITIALS, "7,13,99,1,2,3,4,5,6,8,9,-1\n")
    particle_type, four_vertex, px_py_pz_m, charge = read_initials(path)[7]
    assert particle_type == 13
    assert four_vertex == (1, 2, 3, 4)
    assert px_py_pz_m == (5, 6, 8, 9)
    assert charge == -1