from dataclasses import dataclass
import numpy as np
from pathlib import Path
import h5py
import logging
logger = logging.getLogger(__name__)

from .parameters import Parameters


@dataclass
class HaloCatalog:
    """
    Halo catalog that stores the halo positions and masses at a given redshift snapshot.
    """
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    M: np.ndarray

    def __post_init__(self):
        self.X = np.asarray(self.X)
        self.Y = np.asarray(self.Y)
        self.Z = np.asarray(self.Z)
        self.M = np.asarray(self.M)
        assert self.X.size == self.Y.size and self.X.size == self.Z.size and self.X.size == self.M.size, "Halo catalog arrays must have the same length."


    @property
    def alpha(self) -> np.ndarray:
        """
        Halo mass accretion rate.
        """
        # TODO determine based on the ACTUAL halo mass history
        return np.ones_like(self.M) * 0.79


    def get_halo_indices(self, parameters: Parameters, alpha_index: int, mass_index:int) -> np.ndarray:
        """
        Computes which halos from the current snapshot lie within the mass and alpha range that are specified.
        Returns their indices.
        """
        if self.M.size == 0:
            return []
        
        # +2 since we want 2 values        
        alpha_range = parameters.source.mass_accretion_alpha_range[alpha_index: alpha_index + 2]
        mass_range = parameters.simulation.halo_mass_bins[mass_index: mass_index + 2]
        # Get the indices of the halos that are within the mass and alpha range
        indices_match = np.where(
            (self.M >= mass_range[0]) & (self.M < mass_range[1]) &
            (self.alpha >= alpha_range[0]) & (self.alpha < alpha_range[1])
        )[0] # in this case where returns two arrays, we only want the first one
        logger.debug(
            "alpha_range=%s, mass_range=%s -> %d matches with mean mass=%.2e, mean alpha=%.2e",
            alpha_range, mass_range, len(indices_match), self.M.mean(), self.alpha.mean()
        )
        return indices_match


    def to_mesh(self, parameters: Parameters) -> np.ndarray:
        """
        Converts the halo catalog to a 3D mesh of halo counts using nearest neighbor interpolation.

        Parameters
        ----------
        parameters : Parameters
            Contains simulation parameters like box size and number of cells.

        Returns
        -------
        np.ndarray
            3D numpy array representing the halo count mesh.
        """
        physical_size = parameters.simulation.Lbox
        grid_size = parameters.simulation.Ncell
        mesh = np.zeros((grid_size, grid_size, grid_size))
        # Convert to physical coordinates and map to grid indices
        x = np.clip(np.round(self.X * grid_size / physical_size).astype(int), 0, grid_size - 1)
        y = np.clip(np.round(self.Y * grid_size / physical_size).astype(int), 0, grid_size - 1)
        z = np.clip(np.round(self.Z * grid_size / physical_size).astype(int), 0, grid_size - 1)
        # Efficiently increment the mesh at the halo positions
        np.add.at(mesh, (x, y, z), 1)
        return mesh
    


    ### methods that create new halo catalogs
    def at_indices(self, indices: list) -> "HaloCatalog":
        """
        Returns a new HaloCatalog (copy) containing only a subset of the halos.
        Usually this will be used to select halos that have the same mass / accretion profile
        """
        return HaloCatalog(
            X = self.X[indices],
            Y = self.Y[indices],
            Z = self.Z[indices],
            M = self.M[indices]
        )


    @classmethod
    def load_21cmfast(cls, path: Path, parameters: Parameters) -> "HaloCatalog":        
        with h5py.File(path, 'r') as f:
            try:
                haloes = f['PerturbHaloField']
                # convert to numpy array as an intermediate step
                m, xyz = haloes['halo_masses'], haloes['halo_coords']

                scaling = float(parameters.simulation.Lbox / parameters.simulation.Ncell)

                # 21cmfast quantities need to be rescaled
                return cls(
                    M = np.array(m) * parameters.cosmology.h,
                    X = np.array(xyz[:, 0]), # * scaling,
                    Y = np.array(xyz[:, 1]), # * scaling,
                    Z = np.array(xyz[:, 2]), # * scaling,
                )
            except KeyError:
                raise KeyError("Halo catalog not found in the file.")


    @classmethod
    def load_pkdgrav(cls, path: Path, parameters: Parameters) -> "HaloCatalog":
        catalog = np.loadtxt(path)
        if catalog.shape == (0,):
            catalog = np.ndarray((0, 4))

        # TODO review the rescaling
        return cls(
            M = catalog[:, 0] * parameters.cosmology.h,
            X = catalog[:, 1],# * parameters.simulation.Lbox,
            Y = catalog[:, 2],# * parameters.simulation.Lbox,
            Z = catalog[:, 3],# * parameters.simulation.Lbox,
        )


    @classmethod
    def load(cls, path: Path, parameters: Parameters) -> "HaloCatalog":
        """
        Load halo catalogues from storage. Depending on the type specified in the parameters, it will either load from 21cmFAST or Pkdgrav data.
    
        Parameters
        ----------
        path : str
            Path to the halo catalogue.
        parameters : Parameters
            Parameters object containing cosmological and simulation parameters.

        Returns
        -------
        HaloCatalog
            Halo catalog object.
        """
        if parameters.simulation.dens_field_type == "21cmFAST":
            logger.debug(f"Loading halo catalog from 21cmFAST: {path}")
            catalog = cls.load_21cmfast(path, parameters)
        elif parameters.simulation.dens_field_type == "pkdgrav":
            logger.debug(f"Loading halo catalog from pkdgrav: {path}")
            catalog = cls.load_pkdgrav(path, parameters)
        else:
            raise ValueError(f"Unknown halo catalog type: {parameters.simulation.dens_field_type}. Supported types are: 21cmFAST, pkdgrav.")

        # filter out haloes that are considerd non-star forming
        condition_min = catalog.M > parameters.source.halo_mass_min
        condition_max = catalog.M < parameters.source.halo_mass_max
        indices = np.where(np.logical_and(condition_min, condition_max))
        return catalog.at_indices(indices)
