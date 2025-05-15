"""
Global parameters for this simulation. They encompass the astrophysical parameters of the source, the cosmological parameters, the simulation parameters, the solver parameters, the excursion set parameters, and the halo mass function parameters.
Slots are used to prevent the creation of new attributes. This is useful to avoid typos and to have a clear overview of the parameters.
"""

from pathlib import Path
import importlib
import hashlib
from dataclasses import dataclass, field, is_dataclass, fields
from typing import Literal, Union
import numpy as np


@dataclass(slots = True)
class SourceParameters:
    """
    Parameters for the sources of radiation. Sensible defaults are provided.
    """

    source_type: Literal['SED', 'Ghara', 'Ross', 'constant'] = 'SED'
    """source type. SED, Ghara, Ross, constant"""

    energy_min_sed_xray: int = 500
    """minimum energy of normalization of xrays in eV"""

    energy_max_sed_xray: int = 2000
    """maximum energy of normalization of xrays in eV"""

    energy_cutoff_min_xray: int = 500
    """lower energy cutoff for the xray band"""

    energy_cutoff_max_xray: int = 2000
    """upper energy cutoff for the xray band"""

    alS_xray: float = 1.00001
    """PL sed Xray part N ~ nu**-alS [nbr of photons/s/Hz] #TODO"""

    xray_normalisation: float = 3.4e40
    """Xray normalization [(erg/s) * (yr/Msun)] (astro-ph/0607234 eq22)"""

    n_lyman_alpha_photons: int = 9690
    """number of lyal photons per baryons in stars"""

    lyman_alpha_power_law: float = 0.0
    """power law index for lyal. 0.0 for constant, 1.0 for linear, 2.0 for quadratic"""

    halo_mass_min: float = 1e8
    """Minimum mass of star forming halo. Mdark in HM. Objects below this mass are not considered during the painting process"""

    halo_mass_max: float = 1e16
    """Maximum mass of star forming halo. Objects above this mass are not considered during the painting process"""

    f_st: float = 0.05
    """TODO"""
    Mp: float = 2.8e11 * 0.68
    """TODO"""
    g1: float = 0.49
    """TODO"""
    g2: float = -0.61
    """TODO"""
    Mt: float = 1e8
    """TODO"""
    g3: float = 4
    """TODO"""
    g4: float = -1
    """TODO"""
    Nion: int = 5000
    """TODO"""

    f0_esc: float = 0.2
    """photon escape fraction f_esc = f0_esc * (M/Mp)^pl_esc"""

    Mp_esc: float = 1e10
    """TODO"""
    pl_esc: float = 0.0
    """TODO"""

    min_xHII_value: int = 0
    """Lower limit for the ionization fraction. All pixels with xHII < min_xHII_value will be set to this value."""




@dataclass(slots = True)
class SolverParameters:
    """
    Solver parameters for the simulation.
    """
    redshifts: np.ndarray = field(default_factory=lambda: np.arange(25, 6, -0.5))
    """Array of redshifts for the simulation. This should exactly match the redshifts of the halo catalogs. This should also be monotonically decreasing."""

    fXh: Literal['constant', 'variable'] = 'constant'
    """if fXh is constant here, it will take the value 0.11. Otherwise, we will compute the free e- fraction in neutral medium and take the fit fXh = xe**0.225"""


@dataclass(slots = True)
class SimulationParameters:
    """
    Parameters that are used to run the simulation. These are used in the generation of the halo profiles and when converting the halo profiles to a grid.

    Attributes:
        kmin: Minimum k value.
        kmax: Maximum k value.
        kbin: Either a path to a text file containing kbin edges values or an int (number of bins to measure PS).

        T_saturated: If True, we will assume Tk >> Tcmb.
        reio: If False, we will assume xHII = 0.
    """
    halo_mass_accretion_alpha: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 0.9, 10))
    """Coefficient for exponential mass accretion. Since beorn distinguishes between accretion rates a range should be specified"""

    halo_mass_bin_min: float = 1e5
    """Minimum halo mass bin in solar masses."""

    halo_mass_bin_max: float = 1e14
    """Maximum halo mass bin in solar masses."""

    halo_mass_bin_n: int = 12
    """Number of mass bins."""

    Ncell: int = 128
    """Number of pixels of the final grid. This is the number of pixels in each dimension. The total number of pixels will be Ncell^3."""

    Lbox: int = 100
    """Box length, in [Mpc/h]. This is the length of the box in each dimension. The total volume will be Lbox^3."""

    halo_catalogs: list[Path] = None
    """List of paths to the halo catalogs at each redshift. This should have the same number of elements as the number of redshifts in the solver parameters and be sorted in the same order."""

    density_fields: list[Path] = None
    """List of paths to the density fields. This should have the same number of elements as the number of redshifts in the solver parameters and be sorted in the same order."""

    input_type: Literal['21cmFAST', 'pkdgrav'] = 'pkdgrav'
    """Can be either 21cmFAST or pkdgrav. Depending on the halo input the format and normalization of the fields is handled differently."""

    store_grids: list = ('Tk', 'bubbles', 'lyal', 'dTb')
    """List of the grids to store. Simulating only the needed grids will speed up the simulation. The available grids are: Tk, bubbles, lyal, dTb."""

    cores: int = 1
    """Number of cores used in parallelization. The computation for each redshift can be parallelized with a shared memory approach. This is the number of cores used for this. Keeping the number at 1 disables parallelization."""

    # TODO rename these
    # TODO are these even used?
    kmin: float = 3e-2
    """Minimum value of the k binning used for the power spectrum"""
    kmax: float = 4
    """Maximum value of the k binning used for the power spectrum"""
    kbin: int = 30
    """Number of k bins used for the power spectrum."""

    spreading_pixel_threshold: Union[int, None] = None
    """When spreading the excess ionization fraction, treat all the connected regions with less than "thresh_pixel" as a single connected region (to speed up). If set to None, a default nonzero value will be used"""

    spreading_subgrid_approximation: bool = True
    """When spreading the excess ionization fraction and running distance_transform_edt, whether or not to do the subgrid approximation."""

    minimum_grid_size_heat: int = 4
    """Minimum grid size used when computing the heat kernel from its associated profile."""

    minimum_grid_size_lyal: int = 16
    """Minimum grid size used when computing the lyal kernel from its associated profile."""

    random_seed: int = 12345
    """Random seed for the random number generator. This is used to generate the random numbers for the halo catalogs and the density fields when using 21cmfast."""

    compute_s_alpha_fluctuations: bool = True
    """Whether or not to include the fluctuations in the suppression factor S_alpha when computing the x_al fraction."""

    compute_x_coll_fluctuations: bool = True
    """Whether or not to include the fluctuations in the collisional coupling coefficient x_coll when computing the x_tot fraction."""

    # derived properties that are directly related to the parameters
    @property
    def halo_mass_bins(self) -> np.ndarray:
        return np.logspace(np.log10(self.halo_mass_bin_min), np.log10(self.halo_mass_bin_max), self.halo_mass_bin_n, base=10)    

    @property
    def kbins(self) -> np.ndarray:
        return np.logspace(np.log10(self.kmin), np.log10(self.kmax), self.kbin, base=10)


@dataclass(slots = True)
class CosmologyParameters:
    """
    Cosmological parameters for the simulation.
    
    Attributes:
        Om: Matter density parameter.
        Ob: Baryon density parameter.
        Ol: Dark energy density parameter.
        rho_c: Critical density of the universe.
        h: Dimensionless Hubble parameter.
        sigma_8: Amplitude of the matter power spectrum on 8 Mpc/h scales.
        ns: Scalar spectral index.
        ps: Path to the input Linear Power Spectrum.
        correlation_function: Path where the corresponding correlation function will be stored.
        HI_frac: HI number fraction. Only used when running H_He_Final.
        clumping: Rescale the background density. Set to 1 to get the normal 2h profile term.
        z_decoupling: Redshift at which the gas decouples from CMB and starts cooling adiabatically.
    """
    
    Om: float = 0.31
    Ob: float = 0.045
    Ol: float = 0.68
    rho_c: float = 2.775e11
    h: float = 0.68
    sigma_8: float = 0.83
    ns: float = 0.96
    ps: Path = Path(importlib.util.find_spec('beorn').origin).parent / 'files' / 'PCDM_Planck.dat'
    correlation_function: Path = Path(importlib.util.find_spec('beorn').origin).parent / 'files' / 'corr_fct.dat'
    HI_frac: float = 1 - 0.08
    clumping: int = 1
    z_decoupling: int = 135


@dataclass(slots = True)
class ExcursionSetParameters:
    """
    SemiNumerical Parameters
    
    Attributes:
        R_max: Mpc/h. The scale at which we start the excursion set.
        n_rec: Mean number of recombination per baryon.
        stepping: When doing the exc set, we smooth the field over varying scales. We loop and increase this scale logarithmically (R=1.1*R).
    """
    R_max: float = 40
    n_rec: int = 3
    stepping: float = 1.1


@dataclass(slots = True)
class HaloMassFunctionParameters:
    """
    Parameters related to analytical halo mass function (PS formalism. Used to compute variance in EPS_MAR, and for subhalo MF in excursion set).
    
    Attributes:
        filter: tophat, sharpk or smoothk
        c: scale to halo mass relation (1 for tophat, 2.5 for sharp-k, 3 for smooth-k)
        q: q for f(nu) [0.707,1,1] for [ST,smoothk or sharpk,PS] (q = 0.8 with tophat fits better the high redshift z>6 HMF)
        p: p for f(nu) [0.3,0.3,0] for [ST,smoothk or sharpk,PS]
        delta_c: critical density
        A: A = 0.322 except 0.5 for PS Spherical collapse (to double check)
        m_min: Minimum mass
        m_max: Maximum mass
        Mbin: Number of mass bins
        z: Output redshift values. Should be a list.
    """
    filter: Literal['tophat', 'sharpk', 'smoothk'] = 'tophat'
    c: float = 1
    q: float = 0.85
    p: float = 0.3
    delta_c: float = 1.686
    A: float = 0.322
    m_min: float = 1e4
    m_max: float = 1e16
    Mbin: int = 300
    z: list = field(default_factory=lambda: [0])



@dataclass(slots = True)
class Parameters:
    """
    Group all the parameters for the simulation.

    Attributes:
        source: SourceParameters
        solver: SolverParameters
        cosmology: CosmologyParameters
        simulation: SimulationParameters
        excursion_set: ExcursionSetParameters
        halo_mass_function: HaloMassFunctionParameters
    """
    source: SourceParameters = field(default_factory = SourceParameters)
    solver: SolverParameters = field(default_factory = SolverParameters)
    cosmology: CosmologyParameters = field(default_factory = CosmologyParameters)
    simulation: SimulationParameters = field(default_factory = SimulationParameters)
    excursion_set: ExcursionSetParameters = field(default_factory = ExcursionSetParameters)
    halo_mass_function: HaloMassFunctionParameters = field(default_factory = HaloMassFunctionParameters)


    def unique_hash(self) -> str:
        """
        Generates a unique hash for the current set of parameters. This can be used as a unique key when caching the computations.
        """
        hash_list = []
        # loops over all (even nested) members and replaces them by hashable types
        for f in fields(self):
            key = f.name
            value = getattr(self, key)
            if is_dataclass(value):
                for ff in fields(value):
                    kk = ff.name
                    v = getattr(value, kk)
                    hash_list.append(make_hashable(v))
            else:
                hash_list.append(make_hashable(value))
        return hashlib.md5(str(hash_list).encode()).hexdigest()



def make_hashable(item):
    if isinstance(item, np.ndarray):
        return make_hashable(item.tolist())
    elif isinstance(item, list):
        return str(item)
    elif isinstance(item, dict):
        return frozenset(item.items())
    elif isinstance(item, Path):
        return item.as_posix()
    else:
        return item
