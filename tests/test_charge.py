import numpy as np
import pytest

from charge import LandauDeposit

# 76 e/um most probable, 5 e/um width, 110 um thick sensor
LOC, SCALE, WIDTH = 76.0, 5.0, 110.0


def test_charge_scales_with_thickness():
    deposit = LandauDeposit(LOC, SCALE, WIDTH, seed=0)
    assert deposit.most_probable_charge == pytest.approx(8360.0)
    assert deposit.charge_scale == pytest.approx(550.0)


def test_same_seed_same_charge():
    a = LandauDeposit(LOC, SCALE, WIDTH, seed=1).throw(100)
    b = LandauDeposit(LOC, SCALE, WIDTH, seed=1).throw(100)
    assert np.array_equal(a, b)


def test_different_seed_different_charge():
    a = LandauDeposit(LOC, SCALE, WIDTH, seed=1).throw(100)
    b = LandauDeposit(LOC, SCALE, WIDTH, seed=2).throw(100)
    assert not np.array_equal(a, b)


def test_consecutive_throws_differ():
    deposit = LandauDeposit(LOC, SCALE, WIDTH, seed=0)
    assert not np.array_equal(deposit.throw(10), deposit.throw(10))


def test_external_generator_is_used():
    deposit = LandauDeposit(LOC, SCALE, WIDTH, seed=0)
    a = deposit.throw(10, rng=np.random.default_rng([0, 5]))
    b = deposit.throw(10, rng=np.random.default_rng([0, 5]))
    assert np.array_equal(a, b)


def test_charge_is_never_negative():
    charge = LandauDeposit(LOC, SCALE, WIDTH, seed=3).throw(100_000)
    assert np.all(charge >= 0)


def test_peak_is_at_most_probable_charge():
    deposit = LandauDeposit(LOC, SCALE, WIDTH, seed=4)
    charge = deposit.throw(200_000)
    counts, edges = np.histogram(charge, bins=200, range=(6000, 14000))
    peak = 0.5 * (edges[counts.argmax()] + edges[counts.argmax() + 1])
    assert peak == pytest.approx(deposit.most_probable_charge, abs=150)


def test_long_tail_towards_high_charge():
    charge = LandauDeposit(LOC, SCALE, WIDTH, seed=5).throw(200_000)
    assert np.mean(charge) > np.median(charge)


def test_zero_scale_gives_constant_charge():
    charge = LandauDeposit(LOC, 0.0, WIDTH, seed=0).throw(5)
    assert np.all(charge == pytest.approx(8360.0))


def test_throw_returns_n_values():
    assert LandauDeposit(LOC, SCALE, WIDTH, seed=0).throw(7).shape == (7,)


@pytest.mark.parametrize("loc, scale, width", [(0, 5, 110), (76, -1, 110), (76, 5, 0)])
def test_invalid_parameters(loc, scale, width):
    with pytest.raises(ValueError):
        LandauDeposit(loc, scale, width, seed=0)