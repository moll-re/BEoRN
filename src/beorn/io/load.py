from ..structs.parameters import Parameters
import numpy as np
import h5py

from ..constants import rhoc0

def load_delta_b(parameters: Parameters, index: int):
    """
    Parameters
    ----------
    param:Bunch
    zz : str. Output of fct z_string_format,

    Returns
    ----------
    3D meshgrid of delta_b = rho/mean_rho-1
    """
    
    LBox = parameters.simulation.Lbox
    nGrid = parameters.simulation.Ncell

    if parameters.simulation.density_fields is None:
        raise ValueError('No density fields provided.')
    
    field_path = parameters.simulation.density_fields[index]

    if parameters.simulation.input_type == 'pkdgrav':
        delta_b = load_pkdgrav_density_field(field_path, LBox)

    elif parameters.simulation.input_type == '21cmFAST':
        delta_b = load_21cmfast_density_field(field_path, LBox)

    else:
        raise Exception('param.sim.dens_field_type should be either 21cmFAST or pkdgrav.')

    if nGrid != delta_b.shape[0]:
        # delta_b = reshape_grid(delta_b, nGrid)
        raise ValueError('The density field does not have the same number of pixels as the simulation box.')

    return delta_b


def load_pkdgrav_density_field(file, LBox):
    """
    Parameters
    ----------
    file : String. Path to the pkdgrav density field
    LBox : Float, box size in Mpc/h
    nGrid : Float, number of grid pixels

    Returns
    ----------
    delta = rho_m/rho_mean-1
    3-D mesh grid. Size (nGrid,nGrid,nGrid)
    """
    dens = np.fromfile(file, dtype=np.float32)
    nGrid = round(dens.shape[0]**(1/3))
    pkd = dens.reshape(nGrid, nGrid, nGrid)
    pkd = pkd.T  ### take the transpose to match X_ion map coordinates
    V_total = LBox ** 3
    V_cell = (LBox / nGrid) ** 3
    mass  = (pkd * rhoc0 * V_total).astype(np.float64)
    rho_m = mass / V_cell
    delta_b = (rho_m) / np.mean(rho_m, dtype=np.float64) - 1
    return delta_b


def load_21cmfast_density_field(file, LBox):
    with h5py.File(file, 'r') as f:
        field = f['PerturbedField']
        dens = field['density']

        dens_array = dens[:]
        return dens_array
