import numpy as np


class Readout:
    '''
    @brief Simulates the electronic readout: noise, threshold and noisy pixels
    '''

    def __init__(self, mean_noise, sigma_noise, thr, noisy_fraction, seed):
        '''
        @brief Default constructor
        @param mean_noise : (float) mean of the gaussian noise, in electrons
                            (usually 0, since the pedestal is subtracted)
        @param sigma_noise : (float) sigma of the gaussian noise, in electrons. Must be positive.
        @param thr : (float) threshold in electrons for the signal charge. Must be positive.
        @param noisy_fraction : (float) probability that a pixel fires without
                                any particle in an event. Must be between 0 and 1.
        @param seed : (int) statistical seed for sampling
        '''
        if sigma_noise < 0 or thr < 0:
            raise ValueError("sigma_noise and thr must be non-negative")
        if not 0 <= noisy_fraction < 1:
            raise ValueError("noisy_fraction must be in [0, 1)")

        self.mean_noise = mean_noise
        self.sigma_noise = sigma_noise
        self.thr = thr
        self.noisy_fraction = noisy_fraction
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def roll_noise(self, charge, rng=None):
        '''
        @brief Adds gaussian electronic noise to the charge of each pixel
        @param charge : (array of float) charge of the pixels, in electrons
        @param rng : (numpy.random.Generator, optional) generator to use instead
                     of the one created from seed, e.g. one per event
        @return (array of float) charge with noise, same shape as @p charge
        '''
        rng = self.rng if rng is None else rng
        charge = np.asarray(charge, dtype=float)
        if self.sigma_noise == 0:
            return charge + self.mean_noise
        return charge + rng.normal(self.mean_noise, self.sigma_noise, size=charge.shape)

    def above_threshold(self, charge):
        '''
        @brief Tells which pixels are read out
        @param charge : (array of float) charge of the pixels, in electrons
        @return (array of bool) True where the charge reaches the threshold
        '''
        charge = np.asarray(charge, dtype=float)
        return (charge >= self.thr) & (charge > 0)

    def noisy_pixels(self, modules, grid, rng=None):
        '''
        @brief Pixels fired by noise alone, without particles
        @param modules : (array of int) IDs of the modules where noise is generated
        @param grid : (PixelGrid) pixel matrix of the modules
        @param rng : (numpy.random.Generator, optional) generator to use instead
                     of the one created from seed
        @return geometry_id, channel_0, channel_1, charge : one entry per noisy
                pixel. The number per module follows a Poisson distribution and
                the charge is the threshold plus an exponential with scale
                sigma_noise, so it is always read out.
        '''
        rng = self.rng if rng is None else rng
        modules = np.asarray(modules, dtype=np.uint64)

        mean = self.noisy_fraction * grid.n_0 * grid.n_1
        counts = rng.poisson(mean, size=len(modules))
        total = int(counts.sum())

        geometry_id = np.repeat(modules, counts)
        channel_0 = rng.integers(0, grid.n_0, size=total)
        channel_1 = rng.integers(0, grid.n_1, size=total)
        extra = rng.exponential(self.sigma_noise, size=total) if self.sigma_noise > 0 \
            else np.zeros(total)
        charge = self.thr + extra
        return geometry_id, channel_0, channel_1, charge