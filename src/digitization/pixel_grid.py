import numpy as np


class PixelGrid:
    '''
    @brief Square pixel matrix centred on the origin of the module plane.

    Channel i along an axis covers [min + i * pitch, min + (i + 1) * pitch).
    All lengths are in mm, as in the -measurements.csv files.
    '''

    def __init__(self, pitch, delta_0, delta_1):
        '''
        @brief Builds the matrix.
        @param pitch : (float) pixel pitch in mm, the same along both axes
        @param delta_0 : (float) half-width of the matrix along local_0, in mm
        @param delta_1 : (float) half-width of the matrix along local_1, in mm

        The half-widths are rounded to a whole number of pixels: delta_0 and
        delta_1 store the rounded values actually covered by the matrix.
        '''
        self.pitch = pitch
        self.n_0 = round(2 * delta_0 / pitch)
        self.n_1 = round(2 * delta_1 / pitch)
        self.delta_0 = self.n_0 * pitch / 2
        self.delta_1 = self.n_1 * pitch / 2
        self.min_0 = -self.delta_0
        self.min_1 = -self.delta_1

    def convert_to_channel(self, local_0, local_1):
        '''
        @brief Converts local coordinates on the module plane to channels.
        @param local_0 : (float or array of float) local coordinate along x, in mm
        @param local_1 : (float or array of float) local coordinate along y, in mm
        @return col, row : (int or array of int) channels along local_0 and local_1.
                They can fall outside the matrix: check them with is_in_grid.
        '''
        local_0 = np.asarray(local_0, dtype=float)
        local_1 = np.asarray(local_1, dtype=float)
        col = np.floor((local_0 - self.min_0) / self.pitch).astype(int)
        row = np.floor((local_1 - self.min_1) / self.pitch).astype(int)
        return col, row

    def channel_center(self, channel_0, channel_1):
        '''
        @brief Centre of the given pixels.
        @param channel_0 : (int or array of int) column
        @param channel_1 : (int or array of int) row
        @return center_0, center_1 : (float or array of float) centre in mm
        '''
        channel_0 = np.asarray(channel_0)
        channel_1 = np.asarray(channel_1)
        center_0 = self.min_0 + (channel_0 + 0.5) * self.pitch
        center_1 = self.min_1 + (channel_1 + 0.5) * self.pitch
        return center_0, center_1

    def is_in_grid(self, channel_0, channel_1):
        '''
        @brief Checks whether the given pixels belong to the matrix.
        @param channel_0 : (int or array of int) column
        @param channel_1 : (int or array of int) row
        @return (bool or array of bool) True where the pixel is in the matrix
        '''
        channel_0 = np.asarray(channel_0)
        channel_1 = np.asarray(channel_1)
        return ((channel_0 >= 0) & (channel_0 < self.n_0)
                & (channel_1 >= 0) & (channel_1 < self.n_1))