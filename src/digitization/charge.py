import numpy as np
from scipy.stats import moyal


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