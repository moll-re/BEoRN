"""
The output of the THESAN simulation gives us more information and can be used to infer per-halo accretion rates.
This requires additional processing, which is described in the example/thesan_preprocessing.ipynb notebook.
"""
from pathlib import Path
import h5py
import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..structs.halo_catalog import HaloCatalog
from ..structs.parameters import Parameters
from .alpha_fitting import vectorized_alpha_fit

# TODO - remove hardcoded values!
# from: https://thesan-project.com/thesan/thesan.html
THESAN_PARTICLE_COUNT = 1050**3

# this batch size can easily take up to 50 GB, so be careful
PARTICLE_BATCH_SIZE = 1_000_000_000

MAX_ALPHA_VALUE = 5

def load_tree_cache(cache_file):
    with h5py.File(cache_file, "r") as f:
        tree_halo_ids = f["tree_halo_ids"][:]
        tree_snap_num = f["tree_snap_num"][:]
        tree_mass = f["tree_mass"][:]
        tree_main_progenitor = f["tree_main_progenitor"][:]
    logger.debug(f"Loaded pre-cached THESAN merger tree from: {cache_file}")
    return tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor


def get_halo_accretion_rate_from_tree(parameters: Parameters, current_index: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the (incomplete) accretion rate by performing a fit to the mass history of the halos in the merger tree.
    Since some are not represented in the tree, this function returns the indices and alpha values for the halos that are present in the tree.
    """
    cached_tree = parameters.simulation.halo_catalogs_thesan_tree
    redshift_range = get_lookback_range(parameters, current_index)
    logger.debug(f"Lookback range is {redshift_range}")

    tree_halo_ids, tree_snap_num, tree_mass, tree_main_progenitor = load_tree_cache(cached_tree)

    current_halo_indices = (tree_snap_num == current_index) & (tree_mass > 0)
    current_halo_ids = tree_halo_ids[current_halo_indices]
    current_halo_count = current_halo_indices.sum()


    halo_mass_history = np.ndarray((current_halo_count, redshift_range.size))

    for i in range(redshift_range.size):
        # find the progenitors of the current halos
        progenitor_indices = tree_main_progenitor[current_halo_indices]

        # find the mass of these progenitors
        halo_mass_history[:, i] = tree_mass[progenitor_indices]
        halo_mass_history[progenitor_indices < 0, i] = np.nan  # set negative indices to NaN

        # now we can find the next progenitors
        current_halo_indices = progenitor_indices

    logger.debug(f"Found {np.sum(np.any(halo_mass_history == 0, axis=1))} trees that stopped early")

    # redshift_range = redshifts[snapshot_index - MAX_LOOKBACK + 1:snapshot_index + 1]
    # redshift_range = np.flip(redshift_range)  # flip to match the order of halo_mass_history
    halo_alphas = vectorized_alpha_fit(parameters, redshift_range, halo_mass_history)


    # clip negative alphas to 0 - negative accretion has no meaning in beorn
    halo_alphas = np.clip(halo_alphas, 0, MAX_ALPHA_VALUE)
    return current_halo_ids, halo_alphas


def get_lookback_range(parameters: Parameters, current_index: int) -> np.ndarray:
    # the lookback should cover a fixed time but since the snapshots are spaced in redshift space we need to calculate the range for each redshift
    lookback_time = 200 # Myr
    # depending on the current redshift, the range of redshifts will be longer or shorter
    redshifts = parameters.solver.redshifts
    current_redshift = redshifts[current_index]
    # TODO
    lookback_index = max(0, current_index - 10)

    redshifts = redshifts[lookback_index:current_index + 1]

    return np.flip(redshifts)



def get_halo_information_from_catalog(path: Path, redshift_index: int, parameters: Parameters) -> np.ndarray:
    offset_file = parameters.simulation.halo_catalogs_thesan_offsets[redshift_index]

    current_snapshots = path.rglob("*.hdf5")
    current_snapshots = list(current_snapshots)

    # sort the list according to the number in the filename: 1, 2, 3, ..., 10, 11, ...
    current_snapshots.sort(key=lambda x: int(x.stem.split(".")[1]))

    with h5py.File(offset_file, "r") as f:
        group_offset_indices = f["FileOffsets"]["Group"][:]
        subhalo_start_index = f["FileOffsets"]["Subhalo"][:]

    # the THESAN documentation says that each chunk files shows how many halos/groups the whole snapshot will yield:
    # snapshot_halo_count = f["Header"]["Ngroups_Total"]
    # in practice this key does not exist, so we need the offset file to estimate the number of halos
    snapshot_group_count = int(group_offset_indices[-1]  * 1.5)
    snapshot_subhalo_count = int(subhalo_start_index[-1] * 1.5)

    current_halo_positions = np.zeros((snapshot_group_count, 3))
    current_halo_masses = np.zeros((snapshot_group_count))
    current_subhalo_to_group_mappings = np.zeros((snapshot_group_count), dtype=int)


    group_start_index = 0
    subhalo_start_index = 0
    for i, f in enumerate(current_snapshots):
        with h5py.File(f, "r") as snap:
            # logger.debug(f"Reading group catalog {f.name}...")
            if "GroupPos" not in snap["Group"]:
                # the file is empty because all halos were already loaded
                continue
            group_positions = snap["Group"]["GroupPos"][:]
            group_end_index = group_start_index + group_positions.shape[0]

            current_halo_positions[group_start_index:group_end_index, :] = group_positions
            current_halo_masses[group_start_index:group_end_index] = snap["Group"]["Group_M_Crit200"][:]

            group_start_index = group_end_index

            subhalo_mappings = snap["Subhalo"]["SubhaloGrNr"][:]
            subhalo_end_index = subhalo_start_index + subhalo_mappings.shape[0]
            current_subhalo_to_group_mappings[subhalo_start_index:subhalo_end_index] = subhalo_mappings # no + offset needed apparently
            subhalo_start_index = subhalo_end_index

            # the ids are just the indices in the file plus the offset

    # just keep the ones that were really loaded
    current_halo_positions = current_halo_positions[:group_start_index, :]
    current_halo_masses = current_halo_masses[:group_start_index]
    current_subhalo_to_group_mappings = current_subhalo_to_group_mappings[:subhalo_start_index]

    return current_halo_positions, current_halo_masses, current_subhalo_to_group_mappings



def load_halo_catalog(path: Path, redshift_index: int, parameters: Parameters) -> HaloCatalog:

    current_halo_ids, halo_alphas = get_halo_accretion_rate_from_tree(parameters, redshift_index)

    current_halo_positions, current_halo_masses, current_subhalo_to_group_mappings = get_halo_information_from_catalog(path, redshift_index, parameters)
    snapshot_group_count = current_halo_masses.size

    # join both informations - from the tree (incomplete, but with alpha), and from the group catalog


    # assuming that the indices from the group catalog are strictly monotonic we can force the same sorting on the halo ids obtained from the tree
    sorting = np.argsort(current_halo_ids)

    tree_ids = current_halo_ids[sorting]
    sorted_halo_alphas = halo_alphas[sorting]

    # fill the baseline value and replace it where we have a value: (for the groups!)
    full_alphas = np.ones(snapshot_group_count) * 0.79

    group_ids = current_subhalo_to_group_mappings[tree_ids]
    # fill in the alphas for the halos that are in the tree
    full_alphas[group_ids] = sorted_halo_alphas

    # the alpha fitting returned np.inf for halos that have "too short" mass histories
    # for now we just set them to the baseline value as well, but this should be investigated further
    # TODO
    full_alphas[np.isinf(full_alphas)] = 0.79

    assert full_alphas.shape == current_halo_masses.shape, "The alphas and masses must have the same shape"

    h = parameters.cosmology.h
    catalog = HaloCatalog(
        positions = current_halo_positions * 1e-3 / h, # convert from kpc/h to Mpc/h to Mpc
        masses = current_halo_masses * 10**10 / h, # convert to Msun/h to Msun
        alphas = full_alphas,
        parameters = parameters
    )
    return catalog


def load_density_field(snapshot_path: Path, parameters: Parameters) -> np.ndarray:
    # in this case path is a folder containing the Thesan snapshot files
    # the files have a particular format, cf. https://thesan-project.com/thesan/snapshots.html

    snapshots = snapshot_path.glob("snap_*.hdf5")

    particle_positions = np.zeros((THESAN_PARTICLE_COUNT, 3), dtype=np.float32)

    # load all particles which are spread across multiple file chunks
    start_index = 0
    for snapshot in snapshots:
        # logger.debug(f"Reading snapshot {snapshot.name}...")
        with h5py.File(snapshot, "r") as f:
            positions = f["PartType1"]["Coordinates"][:]
            end_index = start_index + positions.shape[0]
            particle_positions[start_index:end_index, :] = positions
            start_index = end_index

    # map them to a mesh that is LBox x LBox x LBox
    # Create a density mesh from the particle positions
    mesh_size = parameters.simulation.Ncell
    box_size = parameters.simulation.Lbox
    mesh = np.zeros((mesh_size, mesh_size, mesh_size))
    scaling = float(mesh_size / box_size)

    # Convert to physical coordinates and map to grid indices - but do it in batches to avoid memory issues
    for start in range(0, THESAN_PARTICLE_COUNT, PARTICLE_BATCH_SIZE):
        end = min(start + PARTICLE_BATCH_SIZE, THESAN_PARTICLE_COUNT)
        batch_positions = particle_positions[start:end, :] * scaling * 1e-3 / parameters.cosmology.h

        # Clip to ensure indices are within bounds
        x = np.clip(np.round(batch_positions[:, 0]).astype(int), 0, mesh_size - 1)
        y = np.clip(np.round(batch_positions[:, 1]).astype(int), 0, mesh_size - 1)
        z = np.clip(np.round(batch_positions[:, 2]).astype(int), 0, mesh_size - 1)

        # Increment the mesh
        np.add.at(mesh, (x, y, z), 1)

    # Normalize the mesh to get the density field
    delta_b = mesh / np.mean(mesh, dtype=np.float64) - 1
    return delta_b
