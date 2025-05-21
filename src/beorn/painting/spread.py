import numpy as np
from scipy.ndimage import distance_transform_edt
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, wait
from skimage.measure import label

from ..structs.parameters import Parameters

import logging
logger = logging.getLogger(__name__)
MAX_VALUE = 1.0


def spread_excess_ionization(parameters: Parameters, input_grid: np.ndarray):
    """
    TODO
    """
    # to later assert that the count was conserved
    initial_count = np.sum(input_grid)
    if initial_count > np.prod(input_grid.shape) * MAX_VALUE:
        logger.info(f"Grid is fully ionized, no need to redistribute.")
        return np.ones_like(input_grid) * MAX_VALUE

    # Cap values at max_value and calculate excess
    excess_mask = input_grid > MAX_VALUE

    # Label the excess regions
    labeled_excess = label(excess_mask, connectivity=1)
    label_count = np.max(labeled_excess)
    logger.info(f"Found {label_count} excess regions.")

    if label_count == 0:
        logger.debug(f"No excess regions found, no need to redistribute.")
        return input_grid
    
    # Calculate excess values and their indices
    excess_values = np.zeros(label_count)
    excess_indices = np.zeros((label_count, 3), dtype=int)
    for label_idx in range(1, label_count + 1):
        excess_region = labeled_excess == label_idx
        excess_value = np.sum(input_grid[excess_region]) - np.sum(excess_region) * MAX_VALUE
        if excess_value > 0:
            excess_values[label_idx - 1] = excess_value
            excess_indices[label_idx - 1] = np.unravel_index(np.argmax(excess_region), input_grid.shape)

    # Create shared memory for the grid
    grid_shape = input_grid.shape
    grid_dtype = input_grid.dtype
    grid_size = input_grid.size * grid_dtype.itemsize
    shared_mem = shared_memory.SharedMemory(create=True, size=grid_size)
    shared_grid = np.ndarray(grid_shape, dtype=grid_dtype, buffer=shared_mem.buf)
    shared_grid[:] = input_grid  # Copy the grid into shared memory


    # Parallelize the redistribution process
    with ProcessPoolExecutor(max_workers=parameters.simulation.cores) as executor:
        futures = []

        for idx, val in zip(excess_indices, excess_values):
            f = executor.submit(
                process_excess_region,
                shared_mem,
                idx,
                val,
                grid_shape
            )
            futures.append(f)

        completed, uncompleted = wait(futures)
        assert len(uncompleted) == 0, "Some processes did not complete successfully."


    # Retrieve the updated grid from shared memory
    updated_grid = shared_grid.copy()

    # Check if the redistribution was successful
    final_count = np.sum(updated_grid)
    # assert np.isclose(final_count, initial_count), "Redistribution failed: initial and final counts do not match."

    # Clean up shared memory
    shared_mem.close()
    shared_mem.unlink()

    return updated_grid



def process_excess_region(buffer: shared_memory.SharedMemory, index: tuple, excess_value: float, grid_shape):
    """
    Redistribute excess value for a single region.
    """

    logger.debug(f"Processing excess region at index {index} with {excess_value=}")
    # Create a local view of the shared grid
    grid = np.ndarray(grid_shape, buffer=buffer.buf)

    # Define a sub-grid centered around the index
    sub_grid_radius = grid_shape[0] // 3
    x, y, z = index
    x_min, x_max = max(0, x - sub_grid_radius), min(grid_shape[0], x + sub_grid_radius + 1)
    y_min, y_max = max(0, y - sub_grid_radius), min(grid_shape[1], y + sub_grid_radius + 1)
    z_min, z_max = max(0, z - sub_grid_radius), min(grid_shape[2], z + sub_grid_radius + 1)

    # Extract the sub-grid
    sub_grid = grid[z_min:z_max, y_min:y_max, x_min:x_max]
    sub_index = (z - z_min, y - y_min, x - x_min)

    # Calculate distances within the sub-grid
    inverted_sub_grid = np.ones_like(sub_grid)
    inverted_sub_grid[sub_index] = 0
    dist_from_excess = distance_transform_edt(inverted_sub_grid)
    dist_from_excess[sub_grid >= 1] = np.inf  # Exclude already capped regions

    # Sort distances and redistribute excess within the sub-grid
    sorted_indices = np.argsort(dist_from_excess.ravel())
    sorted_coords = np.unravel_index(sorted_indices, sub_grid.shape)

    distributed_value = 0
    index = 0
    max_index = sorted_coords[0].size
    while distributed_value < excess_value and index < max_index:
        coord = (
            sorted_coords[0][index] + z_min,
            sorted_coords[1][index] + y_min,
            sorted_coords[2][index] + x_min,
        )
        # Check that the grid is not already capped at that position
        if grid[coord] < 1:
            available_space = 1 - grid[coord]
            to_add = min(excess_value - distributed_value, available_space)
            grid[coord] += to_add
            distributed_value += to_add
        index += 1

    # If excess_value is not fully distributed, fall back to the full grid
    if distributed_value < excess_value:
        logger.debug(f"Falling back to full grid for redistribution at index {index}.")
        inverted_grid = np.ones_like(grid)
        inverted_grid[index] = 0
        dist_from_excess = distance_transform_edt(inverted_grid)
        dist_from_excess[grid >= 1] = np.inf  # Exclude already capped regions

        sorted_indices = np.argsort(dist_from_excess.ravel())
        sorted_coords = np.unravel_index(sorted_indices, grid_shape)

        while distributed_value < excess_value and index < sorted_coords[0].size:
            coord = (
                sorted_coords[0][index],
                sorted_coords[1][index],
                sorted_coords[2][index],
            )
            if grid[coord] < 1:
                available_space = 1 - grid[coord]
                to_add = min(excess_value - distributed_value, available_space)
                grid[coord] += to_add
                distributed_value += to_add
            index += 1
