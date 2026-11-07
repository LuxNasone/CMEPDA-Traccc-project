import numpy as np
import pytest

from charge import ChargeCloud
from pixel_grid import PixelGrid

PITCH = 0.05  # mm
UM = 1e-3     # mm


@pytest.fixture
def grid():
    return PixelGrid(PITCH, 1.0, 2.0)


def test_charge_is_conserved_far_from_edges(grid):
    _, _, _, value = ChargeCloud(10 * UM).share(grid, [0.012], [0.3], [1000.0])
    assert value.sum() == pytest.approx(1000.0, rel=1e-6)


def test_zero_sigma_fills_one_pixel(grid):
    source, ch0, ch1, value = ChargeCloud(0.0).share(grid, [0.012], [0.3], [500.0])
    col, row = grid.convert_to_channel(0.012, 0.3)
    assert list(source) == [0]
    assert list(ch0) == [col]
    assert list(ch1) == [row]
    assert list(value) == [500.0]


def test_centre_of_pixel_keeps_most_charge(grid):
    x, y = grid.channel_center(20, 40)
    _, ch0, ch1, value = ChargeCloud(10 * UM).share(grid, [x], [y], [1.0])
    best = value.argmax()
    assert (ch0[best], ch1[best]) == (20, 40)
    assert value[best] > 0.9


def test_hit_on_edge_is_shared_equally(grid):
    x_center, y = grid.channel_center(20, 40)
    edge = x_center + PITCH / 2
    _, ch0, ch1, value = ChargeCloud(10 * UM).share(grid, [edge], [y], [1.0])
    left = value[(ch0 == 20) & (ch1 == 40)].sum()
    right = value[(ch0 == 21) & (ch1 == 40)].sum()
    assert left == pytest.approx(right, rel=1e-6)


def test_centroid_is_close_to_hit(grid):
    x, y = 0.0123, -0.4567
    _, ch0, ch1, value = ChargeCloud(30 * UM).share(grid, [x], [y], [1.0])
    center_0, center_1 = grid.channel_center(ch0, ch1)
    assert np.average(center_0, weights=value) == pytest.approx(x, abs=5 * UM)
    assert np.average(center_1, weights=value) == pytest.approx(y, abs=5 * UM)


def test_pixels_outside_grid_are_dropped(grid):
    _, ch0, ch1, value = ChargeCloud(30 * UM).share(grid, [-0.999], [0.0], [1.0])
    assert np.all(grid.is_in_grid(ch0, ch1))
    assert value.sum() < 1.0


def test_source_tells_which_crossing(grid):
    source, _, _, value = ChargeCloud(0.0).share(grid, [0.0, 0.5], [0.0, 0.5], [1.0, 2.0])
    assert list(source) == [0, 1]
    assert list(value) == [1.0, 2.0]


def test_many_crossings_at_once(grid):
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.8, 0.8, 500)
    y = rng.uniform(-1.8, 1.8, 500)
    charge = rng.uniform(1000, 10000, 500)
    source, _, _, value = ChargeCloud(10 * UM).share(grid, x, y, charge)
    totals = np.bincount(source, weights=value, minlength=500)
    assert totals == pytest.approx(charge, rel=1e-6)


def test_no_crossings(grid):
    source, *_ = ChargeCloud(10 * UM).share(grid, [], [], [])
    assert len(source) == 0


def test_invalid_sigma():
    with pytest.raises(ValueError):
        ChargeCloud(-1.0)