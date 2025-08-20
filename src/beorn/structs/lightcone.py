'''
Wrapper class for building lightcones using tools21cm.
'''

import numpy as np
import tools21cm as t2c
from dataclasses import dataclass
from .parameters import Parameters
from .global_profiles import GridDataMultiZ

@dataclass
class Lightcone:
    data: np.ndarray
    redshifts: np.ndarray
    parameters: Parameters
    quantity: str


    @classmethod
    def build(cls, parameters: Parameters, grid: GridDataMultiZ, quantity='dTb') -> "Lightcone":
        # prepare the data to a format readable by tools21cm
        try:
            grid_data = getattr(grid, quantity)
        except KeyError:
            raise ValueError(f"Quantity '{quantity}' not found in grid data.")

        # data_dict = {grid.z[i]: grid_data[i, ...] for i in range(len(grid.z))}
        def reading_function(i):
            gd = grid_data[i, ...]
            # cleanup nans TODO - remove
            gd[np.isnan(gd)] = 0.0
            return gd

        scale_factors = 1 / (grid.z[:] + 1)


        # value_dict = {grid.z[i]: reading_function(i) for i in range(0, len(grid.z), 10)}

        lightcone_data, lightcone_redshifts = t2c.make_lightcone(
            filenames = range(grid.z.size),
            # file_redshifts = grid.z[:],
            file_redshifts = scale_factors,
            reading_function = reading_function,
            los_axis = 2,
            raw_density = False,
            box_length_mpc = parameters.simulation.Lbox,
        )
        # print(lightcone_data.shape, lightcone_redshifts.shape)
        # assert lightcone_redshifts == grid.z, "Redshifts in lightcone do not match grid redshifts."

        return Lightcone(
            parameters = parameters,
            data = lightcone_data,
            redshifts = lightcone_redshifts,
            quantity = quantity,
        )
