import numpy as np
import pytest

from pixel_grid import PixelGrid

PITCH = 0.05  # 50 um, in mm


@pytest.fixture
def grid():
    return PixelGrid(PITCH, 12.0, 40.0)


# Construction

def test_number_of_pixels(grid):
    assert grid.n_0 == 480
    assert grid.n_1 == 1600


def test_grid_is_centred(grid):
    assert grid.min_0 == pytest.approx(-12.0)
    assert grid.min_1 == pytest.approx(-40.0)


def test_half_width_is_rounded_to_whole_pixels():
    grid = PixelGrid(PITCH, 1.01, 1.0)  # 40.4 pixels -> 40
    assert grid.n_0 == 40
    assert grid.delta_0 == pytest.approx(1.0)
    assert grid.min_0 == pytest.approx(-1.0)


# convert_to_channel

def test_origin_is_between_the_two_central_pixels(grid):
    col, row = grid.convert_to_channel(0.0, 0.0)
    assert (col, row) == (240, 800)


def test_pixel_edges(grid):
    col_before, _ = grid.convert_to_channel(-12.0 + PITCH - 1e-6, 0.0)
    col_after, _ = grid.convert_to_channel(-12.0 + PITCH + 1e-6, 0.0)
    assert col_before == 0
    assert col_after == 1


def test_channels_are_integers(grid):
    col, row = grid.convert_to_channel([0.012, 1.5], [-0.46, 3.0])
    assert np.issubdtype(col.dtype, np.integer)
    assert np.issubdtype(row.dtype, np.integer)


def test_accepts_lists_and_arrays(grid):
    col, row = grid.convert_to_channel([0.012, -0.46], np.array([0.3, 0.3]))
    assert col.shape == (2,)
    assert row.shape == (2,)


def test_axes_are_not_swapped(grid):
    col, row = grid.convert_to_channel(-11.99, 39.99)
    assert col == 0
    assert row == grid.n_1 - 1


def test_point_outside_gives_channel_outside(grid):
    col, row = grid.convert_to_channel(-12.01, 40.01)
    assert col == -1
    assert row == grid.n_1


# channel_center

def test_center_of_first_pixel(grid):
    center_0, center_1 = grid.channel_center(0, 0)
    assert center_0 == pytest.approx(-12.0 + PITCH / 2)
    assert center_1 == pytest.approx(-40.0 + PITCH / 2)


def test_center_and_channel_are_consistent(grid):
    channels_0 = np.arange(grid.n_0)
    channels_1 = np.arange(grid.n_0)  # same length, inside both axes
    center_0, center_1 = grid.channel_center(channels_0, channels_1)
    col, row = grid.convert_to_channel(center_0, center_1)
    assert np.array_equal(col, channels_0)
    assert np.array_equal(row, channels_1)


def test_point_is_close_to_its_pixel_center(grid):
    rng = np.random.default_rng(0)
    local_0 = rng.uniform(-12.0, 12.0, 1000)
    local_1 = rng.uniform(-40.0, 40.0, 1000)
    center_0, center_1 = grid.channel_center(*grid.convert_to_channel(local_0, local_1))
    assert np.all(np.abs(center_0 - local_0) <= PITCH / 2 + 1e-12)
    assert np.all(np.abs(center_1 - local_1) <= PITCH / 2 + 1e-12)


# is_in_grid

def test_is_in_grid_with_arrays(grid):
    channels = np.array([-1, 0, grid.n_0 - 1, grid.n_0])
    inside = grid.is_in_grid(channels, np.zeros(4, dtype=int))
    assert list(inside) == [False, True, True, False]


def test_is_in_grid_checks_both_axes(grid):
    assert grid.is_in_grid(0, grid.n_1 - 1)
    assert not grid.is_in_grid(0, grid.n_1)
    assert not grid.is_in_grid(0, -1)


def test_is_in_grid_with_single_values(grid):
    assert grid.is_in_grid(10, 10)