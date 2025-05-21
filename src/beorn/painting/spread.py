import numpy as np
from scipy.ndimage import distance_transform_edt
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, wait
from ..structs.parameters import Parameters

import logging
logger = logging.getLogger(__name__)
MAX_VALUE = 1.0


def spread_excess_ionization(parameters: Parameters, input_grid: np.ndarray):
    # Cap values at max_value and calculate excess
    excess_mask = input_grid > MAX_VALUE
    excess_indices = np.argwhere(excess_mask)
    excess_values = input_grid[excess_mask] - MAX_VALUE
    input_grid[excess_mask] = MAX_VALUE  # Cap the excess regions to MAX_VALUE

    if excess_indices.size == 0:
        return input_grid  # No excess to redistribute
    
    logger.info(f"Redistributing excess ionization: {len(excess_indices)} regions found.")

    initial_count = np.sum(input_grid)
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
                tuple(idx),
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



def process_excess_region(buffer, index, value, grid_shape):
    """
    Redistribute excess value for a single region.
    """
    logger.debug(f"Processing excess region at index {index} with value {value}")
    # Create a local view of the shared grid
    local_grid = np.ndarray(grid_shape, buffer=buffer.buf)

    # Calculate distances from the excess region
    inverted_grid = np.ones_like(local_grid)
    inverted_grid[tuple(index)] = 0
    dist_from_excess = distance_transform_edt(inverted_grid)
    dist_from_excess[local_grid >= 1] = np.inf  # Exclude already capped regions

    # Sort distances and redistribute excess
    sorted_indices = np.argsort(dist_from_excess.ravel())
    sorted_coords = np.unravel_index(sorted_indices, grid_shape)

    cumulative_sum = 0
    index = 0
    while cumulative_sum < value:
        coord = sorted_coords[0][index], sorted_coords[1][index], sorted_coords[2][index]
        if local_grid[coord] < 1:
            available_space = 1 - local_grid[coord]
            to_add = min(value - cumulative_sum, available_space)
            local_grid[coord] += to_add
            cumulative_sum += to_add
        index += 1

    # remove the excess region
    local_grid[tuple(index)] -= value
