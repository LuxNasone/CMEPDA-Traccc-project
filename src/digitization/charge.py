import numpy as np
from scipy.stats import moyal
import math

from scipy.special import ndtr
from pixel_grid import *

class LandauDeposit:
    '''
    @brief Class to simulate the charge released in the sensor according to a Landau distribution (approximated as a Moyal distribution).
    '''

    def __init__(self, loc, scale, width, seed):
        '''
        @brief Default constructor
        @param loc : (float) most probable charge per unit thickness, in electrons/um
        @param scale : (float) width of the distribution per unit thickness, in electrons/um
        @param width : (float) sensor thickness, in um: it scales the total charge
        @param seed : (int) statistical seed of the random generator
        '''
        if width <= 0:
            raise ValueError("width must be positive")
        if loc <= 0 or scale < 0:
            raise ValueError("loc must be positive and scale non-negative")

        self.loc = loc
        self.scale = scale
        self.width = width
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @property
    def most_probable_charge(self):
        '''
        @brief Most probable charge released in the whole sensor, in electrons
        '''
        return self.loc * self.width

    @property
    def charge_scale(self):
        '''
        @brief Width of the distribution for the whole sensor, in electrons
        '''
        return self.scale * self.width

    def throw(self, n=1, rng=None):
        '''
        @brief Given all attributes above determines the charge released
        @param n : (int) number of particles crossing the sensor
        @param rng : (numpy.random.Generator, optional) generator to use instead
                     of the one created from seed, e.g. one per event
        @return (array of float) charge released by each particle, in electrons.
                Negative values are set to 0.
        '''
        rng = self.rng if rng is None else rng
        if self.charge_scale == 0:
            return np.full(n, self.most_probable_charge)

        charge = moyal.rvs(loc=self.most_probable_charge, scale=self.charge_scale,
                           size=n, random_state=rng)
        return np.clip(charge, 0.0, None)

class ChargeCloud:
    '''
    @brief Spreads the deposited charge on the pixels with a gaussian cloud.

    '''

    def __init__(self, sigma, n_sigma=4.0):
        '''
        @brief Default constructor
        @param sigma : (float) width of the cloud, in mm. With 0 all the charge
                       goes to the crossed pixel.
        @param n_sigma : (float) distance, in sigma, beyond which the charge is
                         neglected
        '''
        if sigma < 0 or n_sigma <= 0:
            raise ValueError("sigma must be non-negative and n_sigma positive")
        self.sigma = sigma
        self.n_sigma = n_sigma

    def share(self, grid, local_0, local_1, charge):
        '''
        @brief Shares the charge of each crossing among the nearby pixels
        @param grid : (PixelGrid) pixel matrix of the module
        @param local_0, local_1 : (array of float) crossing points, in mm
        @param charge : (array of float) charge of each crossing, in electrons
        @return source, channel_0, channel_1, value : for each pixel reached, the
                index of the crossing that lit it, its channels and the charge it
                collected. Pixels outside the matrix are dropped.
        '''
        local_0 = np.atleast_1d(np.asarray(local_0, dtype=float))
        local_1 = np.atleast_1d(np.asarray(local_1, dtype=float))
        charge = np.atleast_1d(np.asarray(charge, dtype=float))

        centre_0, centre_1 = grid.convert_to_channel(local_0, local_1)
        channels_0, fraction_0 = self._axis_fractions(grid, local_0, centre_0, 0)
        channels_1, fraction_1 = self._axis_fractions(grid, local_1, centre_1, 1)

        # One entry for each (crossing, pixel along 0, pixel along 1)
        n, w0 = channels_0.shape
        w1 = channels_1.shape[1]
        shape = (n, w0, w1)
        source = np.broadcast_to(np.arange(n)[:, None, None], shape)
        channel_0 = np.broadcast_to(channels_0[:, :, None], shape)
        channel_1 = np.broadcast_to(channels_1[:, None, :], shape)
        value = charge[:, None, None] * fraction_0[:, :, None] * fraction_1[:, None, :]

        keep = (value > 0) & grid.is_in_grid(channel_0, channel_1)
        return source[keep], channel_0[keep], channel_1[keep], value[keep]

    def _axis_fractions(self, grid, local, centre, axis):
        '''
        @brief Pixels near each crossing along one axis and the charge fraction
               each of them collects
        @param grid : (PixelGrid) pixel matrix
        @param local : (array of float) crossing coordinates along the axis, in mm
        @param centre : (array of int) crossed channel along the axis
        @param axis : (int) 0 for local_0, 1 for local_1
        @return channels, fractions : arrays of shape (crossings, pixels)
        '''
        if self.sigma == 0:
            return centre[:, None], np.ones((len(local), 1))

        k = math.ceil(self.n_sigma * self.sigma / grid.pitch)
        channels = centre[:, None] + np.arange(-k, k + 1)

        centers = grid.channel_center(channels, channels)[axis]
        lower = centers - grid.pitch / 2
        upper = centers + grid.pitch / 2
        x = local[:, None]
        fractions = ndtr((upper - x) / self.sigma) - ndtr((lower - x) / self.sigma)
        return channels, fractions