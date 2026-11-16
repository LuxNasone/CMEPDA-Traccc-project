import numpy as np
import pytest

from charge import ChargeCloud, LandauDeposit
from digitizer import Digitizer
from pixel_grid import PixelGrid
from readout import Readout

UM = 1e-3          
MODULE = 7
MOST_PROBABLE = 76.0 * 100.0  
MODULES = np.array([MODULE], dtype=np.uint64)


def make_digitizer(sigma=0.0, scale=0.0, noise=0.0, thr=0.0, fraction=0.0):
    '''Digitizer on a 40 x 80 pixel grid. By default there is no randomness:
    constant charge, no cloud, no noise and no threshold.'''
    grid = PixelGrid(0.05, 1.0, 2.0)
    deposit = LandauDeposit(76.0, scale, 100.0, seed=0)
    cloud = ChargeCloud(sigma)
    readout = Readout(0.0, noise, thr, fraction, seed=0)
    return Digitizer(grid, deposit, cloud, readout)


def run(digitizer, hit0, hit1, hit_module=None, modules=MODULES, rng=None):
    hit_module = np.full(len(hit0), MODULE, dtype=np.uint64) if hit_module is None \
        else np.asarray(hit_module, dtype=np.uint64)
    rng = np.random.default_rng(0) if rng is None else rng
    return digitizer.digitize(np.asarray(hit0, dtype=float), np.asarray(hit1, dtype=float),
                              hit_module, modules, rng)


# Signal

def test_single_hit_gives_single_pixel():
    digitizer = make_digitizer()
    source, module, channel_0, channel_1, value = run(digitizer, [0.012], [0.3])
    col, row = digitizer.grid.convert_to_channel(0.012, 0.3)
    assert list(source) == [0]
    assert list(module) == [MODULE]
    assert (channel_0[0], channel_1[0]) == (col, row)
    assert value[0] == pytest.approx(MOST_PROBABLE)


def test_outputs_have_same_length():
    outputs = run(make_digitizer(sigma=10 * UM), [0.0249, -0.5], [0.3, 1.0])
    assert len({len(array) for array in outputs}) == 1


def test_charge_is_shared_on_pixel_edge():
    digitizer = make_digitizer(sigma=10 * UM)
    x, y = digitizer.grid.channel_center(20, 40)
    *_, value = run(digitizer, [x + 0.025], [y])
    assert len(value) >= 2
    assert value.sum() == pytest.approx(MOST_PROBABLE, rel=1e-6)


def test_source_points_to_its_hit():
    hit0, hit1 = [-0.5, 0.5], [-1.0, 1.0]
    digitizer = make_digitizer()
    source, _, channel_0, channel_1, _ = run(digitizer, hit0, hit1)
    for s, c0, c1 in zip(source, channel_0, channel_1):
        assert (c0, c1) == digitizer.grid.convert_to_channel(hit0[s], hit1[s])


def test_module_follows_its_hit():
    source, module, *_ = run(make_digitizer(), [0.012, 0.5], [0.3, 1.0], hit_module=[7, 8], modules=np.array([7, 8], dtype=np.uint64))
    assert dict(zip(source.tolist(), module.tolist())) == {0: 7, 1: 8}


def test_large_module_ids_are_kept():
    big = 2**63 + 5
    _, module, *_ = run(make_digitizer(), [0.012], [0.3], hit_module=[big], modules=np.array([big], dtype=np.uint64))
    assert module[0] == big


# Pixels hit more than once

def test_same_pixel_is_summed():
    _, _, _, _, value = run(make_digitizer(), [0.012, 0.013], [0.3, 0.3])
    assert len(value) == 1
    assert value[0] == pytest.approx(2 * MOST_PROBABLE)


def test_same_channels_on_different_modules_are_not_merged():
    _, module, *_ = run(make_digitizer(), [0.012, 0.012], [0.3, 0.3], hit_module=[7, 8],
                        modules=np.array([7, 8], dtype=np.uint64))
    assert sorted(module.tolist()) == [7, 8]


def test_merged_pixel_keeps_largest_contribution():
    digitizer = make_digitizer(sigma=10 * UM)
    x, y = digitizer.grid.channel_center(20, 40)
    # hit 0 in the next pixel, close to the shared edge, leaks a little charge into
    # pixel (20, 40); hit 1 at the centre of pixel (20, 40) gives it most of its charge
    source, _, channel_0, channel_1, _ = run(digitizer, [x + 0.03, x], [y, y])
    shared = (channel_0 == 20) & (channel_1 == 40)
    assert source[shared].tolist() == [1]


def test_every_pixel_appears_once():
    rng_hits = np.random.default_rng(1)
    hit0 = rng_hits.uniform(-0.9, 0.9, 300)
    hit1 = rng_hits.uniform(-1.9, 1.9, 300)
    _, module, channel_0, channel_1, _ = run(make_digitizer(sigma=10 * UM), hit0, hit1)
    pixels = set(zip(module.tolist(), channel_0.tolist(), channel_1.tolist()))
    assert len(pixels) == len(module)


# Noisy pixels

def test_noisy_pixels_have_source_minus_one():
    source, module, *_ = run(make_digitizer(noise=100.0, thr=1000.0, fraction=1e-2),
                             [], [], modules=np.array([7, 8], dtype=np.uint64))
    assert len(source) > 0
    assert np.all(source == -1)
    assert set(module.tolist()) <= {7, 8}


def test_noisy_pixels_are_added_to_signal():
    digitizer = make_digitizer(noise=100.0, thr=1000.0, fraction=1e-2)
    source, *_ = run(digitizer, [0.012], [0.3])
    assert 0 in source
    assert -1 in source


def test_noisy_pixel_on_hit_pixel_is_summed():
    # every pixel is noisy: the hit pixel must still appear once, with both charges
    digitizer = make_digitizer(thr=1000.0, fraction=0.999)
    source, module, channel_0, channel_1, value = run(digitizer, [0.012], [0.3])
    col, row = digitizer.grid.convert_to_channel(0.012, 0.3)
    hit = (channel_0 == col) & (channel_1 == row)
    assert hit.sum() == 1
    assert source[hit][0] == 0
    assert value[hit][0] > MOST_PROBABLE


# Noise and threshold

def test_threshold_removes_small_pixels():
    hit0, hit1 = [0.0249], [0.0249]  # close to a pixel corner
    *_, low = run(make_digitizer(sigma=10 * UM, thr=0.0), hit0, hit1)
    *_, high = run(make_digitizer(sigma=10 * UM, thr=3000.0), hit0, hit1)
    assert len(high) < len(low)
    assert np.all(high >= 3000.0)


def test_electronic_noise_changes_charge():
    *_, value = run(make_digitizer(noise=100.0), [0.012], [0.3])
    assert value[0] != pytest.approx(MOST_PROBABLE)
    assert value[0] == pytest.approx(MOST_PROBABLE, abs=600)  # within 6 sigma


# Empty event and reproducibility

def test_empty_event_gives_no_pixels():
    outputs = run(make_digitizer(), [], [])
    assert all(len(array) == 0 for array in outputs)


def test_same_event_generator_same_pixels():
    digitizer = make_digitizer(sigma=10 * UM, scale=5.0, noise=100.0, thr=1000.0,
                               fraction=1e-3)
    hit0, hit1 = np.linspace(-0.9, 0.9, 50), np.linspace(-1.9, 1.9, 50)
    a = run(digitizer, hit0, hit1, rng=np.random.default_rng([4, 1]))
    b = run(digitizer, hit0, hit1, rng=np.random.default_rng([4, 1]))
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_different_events_get_different_charges():
    digitizer = make_digitizer(scale=5.0)
    *_, a = run(digitizer, [0.012], [0.3], rng=np.random.default_rng([4, 1]))
    *_, b = run(digitizer, [0.012], [0.3], rng=np.random.default_rng([4, 2]))
    assert a[0] != b[0]