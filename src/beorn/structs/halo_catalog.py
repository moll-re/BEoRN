from dataclasses import dataclass
import numpy as np
from pathlib import Path
import h5py
import logging
logger = logging.getLogger(__name__)

from .parameters import Parameters
from ..import alpha_fitting





@dataclass
class HaloCatalog:
    """
    Halo catalog that stores the halo positions and masses at a given redshift snapshot.
    """
    positions: np.ndarray
    """Halo positions in 3D space (X, Y, Z coordinates) => shape=(N, 3)"""
    masses: np.ndarray
    """Halo masses in units of Msun => shape=(N,)"""

    parameters: Parameters
    redshift_index: int = 0


    def __post_init__(self):
        self.masses = np.asarray(self.masses)
        assert self.positions.shape[0] == self.masses.size, "Halo catalog arrays must have the same length."


    def alpha(self, x: np.ndarray) -> np.ndarray:
        """
        Halo mass accretion rate.
        """
        redshifts, mass_history = self.mass_history()
        if mass_history is None:
            logger.warning("No mass history available for alpha calculation. Returning zeros.")
            # Default value for alpha if no mass history is available
            return np.ones(x.size) * 0.79

        # Calculate the alpha values by performing an exponential fit with a fixed end value of m0 = mass_history[:, 0]
        # running this for all halos in one call requires a little bit of reshaping
        # alphas = alpha_fitting.alpha_fit(self.parameters, redshifts, mass_history)
        alphas = alpha_fitting.vectorized_alpha_fit(self.parameters, redshifts, mass_history)
        return alphas


    def mass_history(self) -> tuple[np.ndarray, np.ndarray]:
        if self.redshift_index == 0:
            logger.warning("No past snapshots available for alpha calculation. Returning zeros.")
            return None

        # TODO - infer mass accretion history EFFICIENTLY from merger tree data
        # for now we don't do this
        return None

        fudge_factor = 1e-2
        logger.debug(f"Calculating mass history for {self.parameters.source.mass_accretion_lookback} snapshots")
        mass_past = np.zeros((self.size, self.parameters.source.mass_accretion_lookback))
        # separately handle the current snapshot
        mass_past[:, 0] = self.masses


        max_dist = fudge_factor * self.parameters.simulation.Lbox
        # note the flip so that the order is: current -> past snapshots
        for i in range(1, self.parameters.source.mass_accretion_lookback):
            # BIGG TODO
            past_catalog = HaloCatalog.load(self.parameters, redshift_index=self.redshift_index - i)
            # we make both arrays compatible to a (N, M, 3) shape where the last axis now contains the pairwise distances (for all NxM combinations)
            # take the norm along that last axis and obtain the true distances
            # -> minimal distance between the halos in the current snapshot and the previous snapshot means that it is likely the same halo
            # Compute pairwise distances only for close halos using broadcasting in chunks
            # We'll build a sparse matrix where only distances < fudge_factor * Lbox are stored
            # -> minimal distance between the halos in the current snapshot and the previous snapshot means that it is likely the same halo 
            distances = np.linalg.norm(self.positions[:, None, :] - past_catalog.positions[None, :, :], axis=-1)
            # distances has shape (N, M)

            # find all the halos that lie within the fudge factor distance
            indices = np.where(distances < max_dist)
            
            # Use numpy advanced indexing to directly update mass_past
            np.maximum.at(
                mass_past[:, i],
                indices[0],
                past_catalog.masses[indices[1]]
            )

        redshifts = np.flip(self.parameters.solver.redshifts[self.redshift_index - self.parameters.source.mass_accretion_lookback:self.redshift_index])
        return redshifts, mass_past



    @property
    def size(self) -> int:
        """
        Returns the number of halos in the catalog.
        """
        return self.masses.size



    def get_halo_indices(self, alpha_range: list[int], mass_range: list[int]) -> np.ndarray:
        """
        Computes which halos from the current snapshot lie within the mass and alpha range that are specified.
        Returns their indices.
        """
        if self.masses.size == 0:
            return []

        alpha_inf, alpha_sup = alpha_range
        mass_inf, mass_sup = mass_range

        # Get the indices of the halos that are within the mass and alpha range
        alphas = self.alpha(self.masses)
        indices_match = np.where(
            (self.masses >= mass_inf) & (self.masses < mass_sup) &
            (alphas >= alpha_inf) & (alphas < alpha_sup)
        )[0]
        # in this case where returns two arrays, we only want the first one

        if indices_match.size != 0:
            logger.debug(
                "alpha_range=(%.2f, %.2f), mass_range=(%.2e, %.2e) -> %d matches",
                alpha_inf, alpha_sup, mass_inf, mass_sup, len(indices_match)
            )
        return indices_match


    def to_mesh(self) -> np.ndarray:
        """
        Converts the halo catalog to a 3D mesh of halo counts using nearest neighbor interpolation. Attention this representation does not take into account the halo masses. The returned quantity should be interpreted as number density! It is meant to be used when creating a map of halos that lie within a predefined mass bin. 

        Returns
        -------
        np.ndarray
            3D numpy array representing the halo count mesh.
        """
        # TODO - allow CIC mapping as well
        physical_size = self.parameters.simulation.Lbox
        grid_size = self.parameters.simulation.Ncell
        mesh = np.zeros((grid_size, grid_size, grid_size))
        scaling = float(grid_size / physical_size)
        # Convert to physical coordinates and map to grid indices
        x = np.clip(np.round(self.positions[..., 0] * scaling).astype(int), 0, grid_size - 1)
        y = np.clip(np.round(self.positions[..., 1] * scaling).astype(int), 0, grid_size - 1)
        z = np.clip(np.round(self.positions[..., 2] * scaling).astype(int), 0, grid_size - 1)
        # Efficiently increment the mesh at the halo positions
        np.add.at(mesh, (x, y, z), 1)
        return mesh



    ### methods that create new halo catalogs
    def at_indices(self, indices) -> "HaloCatalog":
        """
        Returns a new HaloCatalog (copy) containing only a subset of the halos.
        Usually this will be used to select halos that have the same mass / accretion profile
        """
        if indices.size == 0:
            indices = []
        return HaloCatalog(
            positions = self.positions[indices, :],
            masses = self.masses[indices],
            parameters = self.parameters,
            redshift_index = self.redshift_index
        )


    @classmethod
    def load_21cmfast(cls, path: Path, parameters: Parameters) -> "HaloCatalog":
        with h5py.File(path, 'r') as f:
            try:
                haloes = f['PerturbHaloField']
                # convert to numpy array as an intermediate step
                m, positions = haloes['halo_masses'], haloes['halo_coords']

                scaling = float(parameters.simulation.Lbox / parameters.simulation.Ncell)

                # 21cmfast quantities need to be rescaled
                return cls(
                    masses = np.asarray(m) * parameters.cosmology.h,
                    positions = np.asarray(positions) * scaling, # + parameters.simulation.Lbox / 2
                    parameters = parameters,
                )
            except KeyError:
                raise KeyError("Halo catalog not found in the file.")


    @classmethod
    def load_pkdgrav(cls, path: Path, parameters: Parameters) -> "HaloCatalog":
        catalog = np.loadtxt(path)
        if catalog.shape == (0,):
            catalog = np.ndarray((0, 4))

        return cls(
            masses = catalog[:, 0] * parameters.cosmology.h,
            # shift to center the box
            positions = catalog[:, 1:] + parameters.simulation.Lbox / 2,
            parameters = parameters,
        )


    @classmethod
    def load(cls, parameters: Parameters, redshift_index: int) -> "HaloCatalog":
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
        path = parameters.simulation.halo_catalogs[redshift_index]
        if parameters.simulation.input_type == "21cmFAST":
            logger.debug(f"Loading halo catalog from 21cmFAST: {path}")
            catalog = cls.load_21cmfast(path, parameters)
        elif parameters.simulation.input_type == "pkdgrav":
            logger.debug(f"Loading halo catalog from pkdgrav: {path}")
            catalog = cls.load_pkdgrav(path, parameters)
        else:
            raise ValueError(f"Unknown halo catalog type: {parameters.simulation.input_type}. Supported types are: 21cmFAST, pkdgrav.")

        # Add the catalog metadata
        catalog.redshift_index = redshift_index

        # filter out haloes that are considerd non-star forming
        condition_min = catalog.masses > parameters.source.halo_mass_min
        condition_max = catalog.masses < parameters.source.halo_mass_max
        indices = (condition_min & condition_max).nonzero()[0]
        # print(f"{catalog.masses.mean()=}, {parameters.source.halo_mass_min=}, {parameters.source.halo_mass_max=}")
        return catalog.at_indices(indices)
