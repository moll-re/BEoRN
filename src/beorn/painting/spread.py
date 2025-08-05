import numpy as np
from scipy.ndimage import center_of_mass, label
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, wait
from ..structs.parameters import Parameters

import logging
logger = logging.getLogger(__name__)



#### Constants
# TODO: infer these from the parameters
MAX_VALUE = 1.0
BATCH_SIZE = 50
# Offset to ensure the sub-grid is large enough
SUBGRID_SIZE_OFFSET = 10


#### Helper functions

def subgrid_boundaries(center: tuple[float, float, float], sub_grid_radius: int, grid_shape: tuple[int, int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Calculate the boundaries of a sub-grid centered around a given point.
    Ensures the boundaries are within the grid shape.
    """
    x, y, z = center
    x_range, y_range, z_range = np.array([0, 0], dtype=int), np.array([0, 0], dtype=int), np.array([0, 0], dtype=int)

    for j, j_range in zip((x, y, z), (x_range, y_range, z_range)):
        if j < sub_grid_radius:
            j_range[0] = 0
            j_range[1] = 2 * sub_grid_radius
        elif j > grid_shape[0] - sub_grid_radius:
            j_range[0] = grid_shape[0] - 2 * sub_grid_radius
            j_range[1] = grid_shape[0]
        else:
            j_range[0] = j - sub_grid_radius
            j_range[1] = j + sub_grid_radius

    return (x_range[0], x_range[1]), (y_range[0], y_range[1]), (z_range[0], z_range[1])



def spherical_distance_grid(shape: tuple[int, int, int]):
    """Returns a grid of the distances from the center of the grid."""
    center = np.array(shape) // 2
    grid = np.indices(shape).astype(float)
    grid -= center[:, None, None, None]
    distances = np.linalg.norm(grid, axis=0)
    return distances





def spread_excess_ionization(parameters: Parameters, input_grid: np.ndarray):
    """
    Spreads the excess ionization values in the input grid to ensure that no cell exceeds MAX_VALUE. Modifies the grid in place.
    """
    # to later assert that the count was conserved
    initial_count = np.sum(input_grid)
    if initial_count > np.prod(input_grid.shape) * MAX_VALUE:
        logger.info("Grid is fully ionized, skipping redistribution.")
        input_grid.fill(MAX_VALUE)
        return

    # Label the excess regions
    excess_mask = input_grid > MAX_VALUE
    labeled_regions, label_count = label(excess_mask)

    if label_count == 0:
        logger.debug("No excess regions found, skipping redistribution.")
        return

    logger.info("Found %d excess regions.", label_count)

    labels = np.arange(1, label_count + 1)
    grid_shape = input_grid.shape
    grid_dtype = input_grid.dtype
    grid_size = input_grid.size * grid_dtype.itemsize


    # find the center of each excess region
    # centers = center_of_mass(labeled_regions, labels)
    centers = center_of_mass(input_grid, labeled_regions, labels)

    ## Create shared memory for the grids: input, output and labels
    grid_buffer = SharedMemory(create=True, size=grid_size)
    updated_grid = np.ndarray(grid_shape, dtype=grid_dtype, buffer=grid_buffer.buf)
    # copy the data from the input grid to the shared memory, but already cap it at MAX_VALUE
    updated_grid[:] = np.minimum(input_grid, MAX_VALUE)

    # Calculate the excess values for each labeled region - the excess is the input grid - MAX_VALUE
    # bincount returns counts for labels starting from 0, but our labels start from 1
    excess_values = np.bincount(
        labeled_regions.ravel(),
        weights=(input_grid - MAX_VALUE).ravel(),
        minlength=label_count + 1
    )[1:]
    # logger.debug(f"{labels.shape=}, {excess_values.shape=} ({excess_values.min()=}, {excess_values.max()=})")#, {centers.shape=}")

    # Parallelize the redistribution process
    with ProcessPoolExecutor(max_workers=parameters.simulation.cores) as executor:
        futures = []

        for i in range(label_count):
        # for i in range(10):
            c = centers[i]
            v = excess_values[i]
            f = executor.submit(
                process_excess_region,
                c,
                v,
                grid_buffer,
                grid_shape,
                grid_dtype,
            )
            futures.append(f)
            # logger.debug(f"Submitted process for excess region at index {idx}")

        # even for very large grids spreading should be quick so we set a timeout of 5 minutes
        completed, uncompleted = wait(futures, timeout = 300)
        assert len(uncompleted) == 0, f"{len(uncompleted)} processes did not complete successfully."


    input_grid[:] = updated_grid[:]
    # Clean up shared memory
    grid_buffer.close()
    grid_buffer.unlink()

    # Check if the redistribution was successful
    final_count = np.sum(input_grid)
    # assert np.isclose(final_count, initial_count), "Redistribution failed: initial and final counts do not match."
    deviation = (final_count - initial_count) / initial_count
    logger.debug(f"Photon numbers: {initial_count=:.2e}, {final_count=:.2e} => {deviation=:.2e}")



def process_excess_region(center: tuple[float, float, float], excess_value: float, buffer: SharedMemory, shape, dtype) -> None:
    """
    Redistribute excess value for a single region over the shared buffer.
    """
    input_grid = np.ndarray(shape, dtype=dtype, buffer=buffer.buf)

    ## Since we don't want to compute the distance transform for the entire grid, we can limit the area of interest
    # But the subgrid needs to be large enough to contain the excess region (that will be spherically spread)
    grid_size_min = np.sqrt(3) * (excess_value / MAX_VALUE) ** (1/3) + SUBGRID_SIZE_OFFSET
    sub_grid_radius = np.minimum(grid_size_min, shape[0]).astype(int)
    # logger.debug(f"Subgrid radius: {sub_grid_radius}")

    (x_min, x_max), (y_min, y_max), (z_min, z_max) = subgrid_boundaries(center, sub_grid_radius, shape)
    # logger.debug(f"Sub-grid bounds: x({x_min}, {x_max}), y({y_min}, {y_max}), z({z_min}, {z_max})")

    # Define a sub-grid centered around the index with the correct values
    offset = np.array([x_min, y_min, z_min], dtype=int)
    sub_grid = input_grid[x_min:x_max, y_min:y_max, z_min:z_max]

    # logger.debug(f"Processing excess region at {center} with {excess_value=:.2f} => {sub_grid.shape=}")

    # Calculate distances within the sub-grid
    # since we know the center point we can directly compute the distance from the center
    dist_from_excess = spherical_distance_grid(sub_grid.shape)
    # Exclude already capped regions
    dist_from_excess[sub_grid >= MAX_VALUE] = np.inf

    # Sort distances and redistribute excess within the sub-grid
    sorted_indices = np.argsort(dist_from_excess.ravel())
    sorted_coords = np.column_stack(np.unravel_index(sorted_indices, sub_grid.shape))

    distributed_value = 0
    loop_index = 0
    max_index = sorted_coords.shape[0]

    # distribute the excess value in batches to vectorize the operation
    while distributed_value < excess_value and loop_index < max_index:
        # at that stage input_grid is already capped at MAX_VALUE, so we can safely compute the available space

        # create indices for the current batch
        batch_end = min(loop_index + BATCH_SIZE, max_index)
        batch_coords = sorted_coords[loop_index:batch_end, :] + offset
        batch_x, batch_y, batch_z = batch_coords[:, 0], batch_coords[:, 1], batch_coords[:, 2]

        # Calculate the "capacity" of each cell and the remaining excess
        space_per_cell = MAX_VALUE - input_grid[batch_x, batch_y, batch_z]
        # dividing by BATCH_SIZE ensures we don't exceed the initial excess value but means the last batch is not guaranteed to be full up to MAX_VALUE
        excess_per_cell = (excess_value - distributed_value) / BATCH_SIZE
        to_add = np.minimum(excess_per_cell, space_per_cell)
        input_grid[batch_x, batch_y, batch_z] += to_add
        distributed_value += np.sum(to_add)
        loop_index = batch_end
