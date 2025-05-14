"""
Basic functions to load and save profiles, 3D maps etc...
"""

import numpy as np
import os
import tools21cm as t2c
import pickle
import h5py
from .constants import rhoc0, Tcmb0
from .structs.parameters import Parameters



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
    field_paths = parameters.simulation.density_fields

    if parameters.simulation.dens_field_type == 'pkdgrav':
        if field_paths is None:
            raise ValueError('No density fields provided.')
        delta_b = load_pkdgrav_density_field(field_paths[index], LBox)

    elif parameters.simulation.dens_field_type == '21cmFAST':
        delta_b = load_21cmfast_density_field(field_paths[index], LBox)

    elif parameters.simulation.dens_field_type == 'array':
        delta_b = np.loadtxt(field_paths[index])
    else:
        print('param.sim.dens_field_type should be either 21cmFAST or pkdgrav.')

    if nGrid != delta_b.shape[0]:
        delta_b = reshape_grid(delta_b, nGrid)

    return delta_b



def reshape_grid(grid, N):
    """
    Parameters
    ----------
    grid : (a,a,a) a 3D meshgrid.
    new_shape : int. the nbr of pixel per grid for the reshaped array

    Returns
    ----------
    3D meshgrid of shape (N,N,N)
    """
    N_ini = grid.shape[0]

    if (N_ini/N) % 1 != 0 and (N/N_ini) % 1 != 0 :
        print('Your param.sim.Ncell should be a mutiple of a divider of your input density field shape.')
        exit()

    else :
        new_shape = (N, N, N)
        if N < N_ini:
            print('Downsampling the density field to a shape ({},{},{})'.format(N, N, N))
            # Downsample by taking the mean of block_size x block_size x block_size blocks
            block_size = int(N_ini / N)
            # Reshape grid into blocks and take the mean
            arr2 = grid.reshape(new_shape[0], block_size, new_shape[1], block_size, new_shape[2], block_size).mean(axis=(1, 3, 5))

        else:
            print('Oversampling the density field to a shape ({},{},{})'.format(N, N, N))
            # Create arr2 by indexing and expanding grid
            arr2 = grid[np.arange(new_shape[0])[:, None, None] // 2, np.arange(new_shape[1])[None, :, None] // 2, np.arange(
            new_shape[2])[None, None, :] // 2]

    return  arr2

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.argmin(np.abs(array - value))
    return array[idx], idx




def Beta(zz, PS, qty='Tk'):
    if qty == 'Tk':
        Tcmb = Tcmb0 * (1 + zz)
        beta_T = Tcmb / (PS['Tk'] - Tcmb)
        return beta_T
    elif qty == 'lyal':
        x_al = PS['x_al']
        x_tot = x_al + PS['x_coll']
        return x_al / x_tot / (1 + x_tot)
    elif qty == 'reio':
        return -PS['x_HII'] / (1 - PS['x_HII'])
    else:
        print('qty should be either Tk, lyal, or reio.')






def delta_fct(grid):
    """
    grid : np.array, meshgrid.
    returns : grid/mean(grid)-1
    """
    return grid / np.mean(grid) - 1





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



def pixel_position(X,Y,Z,LBox,nGrid):
    """
    Parameters
    ----------
    X,Y,Z : floats, positions in cMpc/h
    LBox : Float, box size in Mpc/h
    nGrid : Float, number of grid pixels

    Returns
    ----------
    Coordinates expressed in grid pixel unit (between 0 and nGrid-1)
    """
    Pos_Halos = np.vstack((X,Y,Z)).T  # Halo positions.
    Pos_Halos_Grid = np.array([Pos_Halos / LBox * nGrid]).astype(int)[0]%nGrid
    return Pos_Halos_Grid




def smooth_field(field,Rsmoothing,Lbox, nGrid):
    """
    Parameters
    ----------
    field : 3d meshgrid with nGrid pixel per dim, in box size Lbox (Mpc/h).
    Lbox : float, box size in Mpc/h
    nGrid : int, number of grid pixels per dim
    Rsmoothing : float (Mpc/h), smoothing scale

    Returns
    ----------
    smoothed_field over a tophat kernel with radius Rsmoothing
    """

    from .excursion_set import profile_kern
    from astropy.convolution import convolve_fft

    x = np.linspace(-Lbox / 2, Lbox / 2, nGrid)  # y, z will be the same.
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    kern = profile_kern(rgrid, Rsmoothing)
    del rgrid
    smoothed_field = convolve_fft(field, kern, boundary='wrap', normalize_kernel=True, allow_huge=True)
    del kern
    del field

    return smoothed_field




def initialise_mpi4py(parameters: Parameters):
    """
    Parameters
    ----------
    Will read in param the number of cores to use.

    Returns
    ----------
    Initialise the mpi4py parallelisation. Returns the rank, size, and com.
    """

    if parameters.simulation.cores > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    return comm, rank, size

def Barrier(comm):
    """
    Parameters
    ----------
    comm : Either None, either MPI.COMM_WORLD.

    Returns
    ----------
    Just to avoid error when running BEoRN on a laptop without mpi4py
    """
    if comm is not None:
        comm.Barrier()




def format_grid_for_PS_measurement(Grid_Temp,Grid_xHII,Grid_xal,nGrid) :
    """
    Parameters
    ----------
    Grid_Temp,Grid_xHII,Grid_xal : the grids as we store them.
    nGrid : param.code.Ncell. Nbr of grid pixel per dim.

    Returns
    ----------
    If a grid is just a number (e.g. xHII = np.array([1]) when the whole universe is ionised), returns an array of one.
    This is to measure power and crosses spectra..
    """

    if Grid_Temp.size == 1:  ## to avoid error when measuring power spectrum
        Grid_Temp = np.full((nGrid, nGrid, nGrid), 1)
    if Grid_xHII.size == 1:
        if Grid_xHII == np.array([0]):
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 0)  ## to avoid div by zero
        elif Grid_xHII == np.array([0]):
            Grid_xHII = np.full((nGrid, nGrid, nGrid), 1)  ## to avoid div by zero
    if Grid_xal.size == 1:
        Grid_xal = np.full((nGrid, nGrid, nGrid), 0)
    return Grid_Temp,Grid_xHII,Grid_xal


