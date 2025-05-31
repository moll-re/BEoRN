import numpy as np
import logging
logger = logging.getLogger(__name__)

from .structs.parameters import Parameters

def exponential_mass_accretion(z, m0, alpha, z0):
    """
    Exponential mass accretion function. Can be called with floats or numpy arrays (in a vectorized way).

    Parameters
    ----------
    z: float or np.ndarray
        Redshift value(s) at which to evaluate the mass accretion.
    m0: float or np.ndarray
        Initial mass at redshift z0.
    alpha: float or np.ndarray
        Exponential growth rate.
    z0: float
        Reference redshift at which m0 is defined.
    """
    # print(f"{m0.shape=}, {alpha.shape=}, {z.shape=}")
    return m0[None, :] * np.exp(alpha[None, :] * (z0 - z[:, None]))




# def nd_fit(f: callable, x: np.ndarray, y: np.ndarray, m0: np.ndarray, z0: float):
#     """
#     Fit a callable function to data using curve_fit, allowing for multi-dimensional inputs.
    
#     Parameters
#     ----------
#     callable : callable
#         The function to fit, which should accept x and parameters as inputs.
#     x : np.ndarray
#         Input data for the independent variable(s).
#     y : np.ndarray
#         Input data for the dependent variable(s).
#     *args : tuple
#         Additional arguments to pass to the callable.
#     **kwargs : dict
#         Keyword arguments to pass to curve_fit.
    
#     Returns
#     -------
#     popt : np.ndarray
#         Optimal values for the parameters of the fit.
#     """
#     halo_number = y.shape[0]
#     redshift_number = y.shape[1]

#     def patched_f(x:np.ndarray, *params):
#         x = x[:redshift_number]
#         alphas = np.array(params)
#         return f(x, m0, alphas, z0).flatten()
    
#     alphas_start_value = np.ones(halo_number) * 0.79
#     alphas_range = (np.zeros(halo_number), np.ones(halo_number) * 1)
#     x = np.hstack([x] * halo_number).T  # Repeat x for each halo

#     print(x.flatten().shape, y.flatten().shape)
#     return curve_fit(patched_f, x.flatten(), y.flatten(), p0=alphas_start_value, bounds=alphas_range)


# def alpha_fit(parameters: Parameters, redshifts: np.ndarray, mass_history: np.ndarray) -> np.ndarray:
#     """
#     Fit the mass accretion history of multiple halos to an exponential function, each with a different initial mass and growth rate.
#     """
#     if mass_history.ndim != 2:
#         raise ValueError("mass_history must be a 2D array with shape (N, M), where N is the number of halos and M is the number of snapshots.")

#     # Get the initial masses at the first snapshot
#     m0 = mass_history[:, 0]
#     z0 = redshifts[0]
#     logger.debug(f"Fitting alpha for {m0.size} halos ({m0.mean()=:.2e}) at {z0=}")
#     popt, _ = nd_fit(
#         exponential_mass_accretion,
#         redshifts,
#         mass_history,
#         m0 = m0,
#         z0 = z0
#     )
#     return popt


# exponential_mass_accretion(z, m0, alpha, z0):
#     """
#     Exponential mass accretion function. Can be called with floats or numpy arrays (in a vectorized way).

#     Parameters
#     ----------
#     z: float or np.ndarray
#         Redshift value(s) at which to evaluate the mass accretion.
#     m0: float or np.ndarray
#         Initial mass at redshift z0.
#     alpha: float or np.ndarray
#         Exponential growth rate.
#     z0: float
#         Reference redshift at which m0 is defined.
#     """
#     # print(f"{m0.shape=}, {alpha.shape=}, {z.shape=}")
#     return m0[None, :] * np.exp(alpha[None, :] * (z0 - z[:, None]))




# def nd_fit(f: callable, x: np.ndarray, y: np.ndarray, m0: np.ndarray, z0: float):
#     """
#     Fit a callable function to data using curve_fit, allowing for multi-dimensional inputs.
    
#     Parameters
#     ----------
#     callable : callable
#         The function to fit, which should accept x and parameters as inputs.
#     x : np.ndarray
#         Input data for the independent variable(s).
#     y : np.ndarray
#         Input data for the dependent variable(s).
#     *args : tuple
#         Additional arguments to pass to the callable.
#     **kwargs : dict
#         Keyword arguments to pass to curve_fit.
    
#     Returns
#     -------
#     popt : np.ndarray
#         Optimal values for the parameters of the fit.
#     """
#     halo_number = y.shape[0]
#     redshift_number = y.shape[1]

#     def patched_f(x:np.ndarray, *params):
#         x = x[:redshift_number]
#         alphas = np.array(params)
#         return f(x, m0, alphas, z0).flatten()
    
#     alphas_start_value = np.ones(halo_number) * 0.79
#     alphas_range = (np.zeros(halo_number), np.ones(halo_number) * 1)
#     x = np.hstack([x] * halo_number).T  # Repeat x for each halo

#     print(x.flatten().shape, y.flatten().shape)
#     return curve_fit(patched_f, x.flatten(), y.flatten(), p0=alphas_start_value, bounds=alphas_range)


# def alpha_fit(parameters: Parameters, redshifts: np.ndarray, mass_history: np.ndarray) -> np.ndarray:
#     """
#     Fit the mass accretion history of multiple halos to an exponential function, each with a different initial mass and growth rate.
#     """
#     if mass_history.ndim != 2:
#         raise ValueError("mass_history must be a 2D array with shape (N, M), where N is the number of halos and M is the number of snapshots.")

#     # Get the initial masses at the first snapshot
#     m0 = mass_history[:, 0]
#     z0 = redshifts[0]
#     logger.debug(f"Fitting alpha for {m0.size} halos ({m0.mean()=:.2e}) at {z0=}")
#     popt, _ = nd_fit(
#         exponential_mass_accretion,
#         redshifts,
#         mass_history,
#         m0 = m0,
#         z0 = z0
#     )
#     return popt


# def objective(params, x, data):
#     """Calculate total residual for fits of to the exponential mass accretion function."""


#     residual = data - exponential_mass_accretion(x, params)
#     ndata, _ = data.shape
#     resid = 0.0*data[:]

#     # make residual per data set
#     for i in range(ndata):
#         resid[i, :] = data[i, :] - gauss_dataset(params, i, x)

#     # now flatten this to a 1D array, as minimize() needs
#     return resid.flatten()



# def vectorized_alpha_fit(parameters: Parameters, redshifts: np.ndarray, mass_history: np.ndarray):
#     """
#     Fit the mass accretion history of multiple halos to an exponential function using lmfit, vectorized for all halos.

#     Parameters
#     ----------
#     redshifts : np.ndarray
#         1D array of redshift values (length M).
#     mass_history : np.ndarray
#         2D array of shape (N, M), where N is the number of halos and M is the number of snapshots.

#     Returns
#     -------
#     result_params : np.ndarray
#         Array of best-fit alpha values for each halo.
#     """
#     n_halos = mass_history.shape[0]
#     z0 = redshifts[0]
#     m0s = mass_history[:, 0]
#     alphas = np.zeros(n_halos)
    
#     model = lm.Model(exponential_mass_accretion, independent_vars=['z'])

#     params = model.make_params()
#     fit_params = lm.Parameters()
#     for i in range(n_halos):
#         fit_params.add(f'alpha_{i}', value=0.79, vary=True, min=0, max=10)

#     lm.minimize(
#         model.fit,
#         params=fit_params,
#         x=redshifts,
#         y=mass_history
#         )
    


def vectorized_alpha_fit(parameters: Parameters, redshifts: np.ndarray, mass_history: np.ndarray) -> np.ndarray:
    """
    Vectorized fitting of alpha for all halos using linear regression in log-space.
    redshifts: (M,) array
    mass_history: (N, M) array
    Returns: (N,) array of best-fit alpha for each halo
    """
    z0 = redshifts[0]
    m0 = mass_history[:, 0]
    dz = z0 - redshifts
    assert np.all(dz <= 0), "Redshift values must be ascending now -> past."
    assert np.all(mass_history >= 0), "Mass history must be non-negative."
    # 0 values are allowed and mean that the halo was not traceable - these should be treated with the baseline case: alpha = 0.79

    # Avoid log(0) or negative masses
    # TODO - make this an assert instead => <= 0 masses are not allowed
    mask = (mass_history > 0) & (m0[:, None] > 0)
    log_mass = np.where(mask, np.log(mass_history), np.nan)
    log_m0 = np.where(m0 > 0, np.log(m0), np.nan)[:, None]
    y = log_mass - log_m0

    # Linear regression: y = alpha * dz
    # alpha = sum(y * dz) / sum(dz^2) for each halo
    numerator = np.nansum(y * dz, axis=1)
    denominator = np.nansum(dz**2)
    alphas = numerator / denominator

    # replace NaNs with the baseline value of 0.79
    alphas = np.where(np.isnan(alphas), 0.79, alphas)
    return alphas
