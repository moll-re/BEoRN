"""TODO"""
from dataclasses import dataclass
import numpy as np
from .base_struct import BaseStruct



@dataclass(slots = True)
class SnapshotQuantities(BaseStruct):
    """
    Class representing spectral quantities for a single redshift snapshot.
    """
    z: float
    # power spectrum scalings
    # TODO - these are implicit from the parameters
    k_bins: np.ndarray

    # actual quantities
    # TODO - rename these to be more descriptive
    PS_dTb: np.ndarray
    PS_dTb_RSD: np.ndarray
    PS_dTb_no_reio: np.ndarray
    PS_dTb_T_sat: np.ndarray
    T_spin: float
    dTb_RSD_mean: float


    third_order_power_spectra: dict[str, np.ndarray] = None
    fourth_order_power_spectra: dict[str, np.ndarray] = None
