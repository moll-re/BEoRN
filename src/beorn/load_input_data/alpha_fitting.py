import numpy as np
import logging
logger = logging.getLogger(__name__)

from ..structs.parameters import Parameters


def vectorized_alpha_fit(parameters: Parameters, redshifts: np.ndarray, mass_history: np.ndarray) -> np.ndarray:
    """
    Vectorized fitting of alpha for all halos using linear regression in log-space.
    redshifts: (M,) array
    mass_history: (N, M) array
    Returns: (N,) array of best-fit alpha for each halo
    """
    z0 = redshifts[0]
    dz = z0 - redshifts
    assert np.all(dz <= 0), "Redshift values must be ascending (now -> past)."
    assert np.all(mass_history >= 0), "Mass history must be non-negative."

    log_mass = np.log(mass_history)
    log_m0 = log_mass[:, 0]
    y = log_mass - log_m0[:, np.newaxis]

    # Linear regression: y = alpha * dz
    # alpha = sum(y * dz) / sum(dz^2) for each halo
    numerator = np.sum(y * dz, axis=1)
    denominator = np.sum(dz**2)
    alphas = numerator / denominator

    return alphas
