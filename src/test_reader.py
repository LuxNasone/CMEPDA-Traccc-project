from file_reader import read_hit

HEADER = "particle_id,geometry_id,tx,ty,tz,tt,tpx,tpy,tpz,te,deltapx,deltapy,deltapz,deltae,index\n"

def write_csv(tmp_path, righe):
    path = tmp_path / "hits.csv"
    path.write_text(HEADER + righe)
    return path

def test_cluster_by_particle(tmp_path):
    path = write_csv(tmp_path,
        "0,100,1,2,3,4,5,6,7,0,0,0,0,0,0\n"
        "0,101,8,9,10,11,12,13,14,0,0,0,0,0,0\n"
        "1,100,15,16,17,18,19,20,21,0,0,0,0,0,0\n")
    hits = read_hit(path)
    assert set(hits) == {0, 1}
    assert len(hits[0]) == 2
    assert len(hits[1]) == 1

def test_right_column(tmp_path):
    path = write_csv(tmp_path, "0,100,1,2,3,4,5,6,7,0,0,0,0,0,0\n")
    four_point, four_impulse = read_hit(path)[0][0]
    assert four_point == (1, 2, 3, 4)
    assert four_impulse == (5, 6, 7, 0)

def test_not_consecutive_hits(tmp_path):
    path = write_csv(tmp_path,
        "0,100,1,2,3,4,5,6,7,0,0,0,0,0,0\n"
        "1,100,8,9,10,11,12,13,14,0,0,0,0,0,0\n"
        "0,101,15,16,17,18,19,20,21,0,0,0,0,0,0\n")
    assert len(read_hit(path)[0]) == 2