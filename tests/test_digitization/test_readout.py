import numpy as np
import pytest

from pixel_grid import PixelGrid
from readout import Readout


@pytest.fixture
def grid():
    return PixelGrid(0.05, 1.0, 2.0)  # 40 x 80 pixels


def make_readout(mean=0.0, sigma=100.0, thr=1000.0, fraction=0.0, seed=0):
    return Readout(mean, sigma, thr, fraction, seed)


# roll_noise

def test_no_noise_keeps_charge():
    charge = np.array([500.0, 1500.0])
    assert np.array_equal(make_readout(sigma=0.0).roll_noise(charge), charge)


def test_noise_statistics():
    noisy = make_readout(mean=10.0, sigma=100.0).roll_noise(np.zeros(200_000))
    assert noisy.mean() == pytest.approx(10.0, abs=1.0)
    assert noisy.std() == pytest.approx(100.0, rel=0.01)


def test_noise_keeps_shape():
    assert make_readout().roll_noise(np.ones((3, 4))).shape == (3, 4)


def test_noise_is_reproducible():
    a = make_readout(seed=1).roll_noise(np.zeros(10))
    b = make_readout(seed=1).roll_noise(np.zeros(10))
    assert np.array_equal(a, b)


def test_external_generator_is_used():
    readout = make_readout()
    a = readout.roll_noise(np.zeros(10), rng=np.random.default_rng([0, 3]))
    b = readout.roll_noise(np.zeros(10), rng=np.random.default_rng([0, 3]))
    assert np.array_equal(a, b)


# above_threshold

def test_threshold():
    mask = make_readout(thr=1000.0).above_threshold([999.9, 1000.0, 5000.0])
    assert list(mask) == [False, True, True]


def test_zero_threshold_drops_empty_pixels():
    mask = make_readout(thr=0.0).above_threshold([-5.0, 0.0, 1.0])
    assert list(mask) == [False, False, True]


# noisy_pixels

def test_no_noisy_pixels_when_disabled(grid):
    geometry_id, *_ = make_readout(fraction=0.0).noisy_pixels([1, 2, 3], grid)
    assert len(geometry_id) == 0


def test_number_of_noisy_pixels(grid):
    modules = np.arange(2000)
    geometry_id, *_ = make_readout(fraction=1e-2).noisy_pixels(modules, grid)
    expected = 1e-2 * grid.n_0 * grid.n_1 * len(modules)
    assert len(geometry_id) == pytest.approx(expected, rel=0.05)


def test_noisy_pixels_are_inside_grid(grid):
    _, channel_0, channel_1, _ = make_readout(fraction=1e-2).noisy_pixels(
        np.arange(500), grid)
    assert np.all(grid.is_in_grid(channel_0, channel_1))


def test_noisy_pixels_belong_to_given_modules(grid):
    modules = np.array([2**63 + 5, 42], dtype=np.uint64)
    geometry_id, *_ = make_readout(fraction=0.1).noisy_pixels(modules, grid)
    assert set(geometry_id.tolist()) <= {2**63 + 5, 42}


def test_noisy_pixels_pass_threshold(grid):
    readout = make_readout(fraction=1e-2)
    *_, charge = readout.noisy_pixels(np.arange(100), grid)
    assert np.all(readout.above_threshold(charge))


# Constructor

@pytest.mark.parametrize("sigma, thr, fraction", [(-1, 1000, 0), (100, -1, 0), (100, 1000, 1.5)])
def test_invalid_parameters(sigma, thr, fraction):
    with pytest.raises(ValueError):
        Readout(0.0, sigma, thr, fraction, seed=0)