import logging
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm

from fragment_feature_detection.decomposition import fit_nmf_matrix_custom_init

from ..config import Config, Constants
from ..fitpeaks import fit_gaussian_elution, least_squares_with_l1_bounds
from ..utils import fraction_explained_variance
from .gpfrun import GPFRun
from .ms1feature import MS1Feature
from .msrun import MSRun
from .scanwindow import ScanWindow

logger = logging.getLogger(__name__)


def fit_scanwindow(
    w: ScanWindow,
    **fitpeaks_kwargs: Dict[str, Any],
) -> None:
    """Fit Gaussian peaks to components in a scan window.

    Args:
        w (ScanWindow): Scan window object to fit peaks to.
        **fitpeaks_kwargs: Keyword arguments passed to fit_gaussian_elution.
    """
    mus = []
    sigmas = []
    maes = []
    keep = []
    for i, s in enumerate(w.component_indices):
        try:
            (mu, s), pcov, mae = fit_gaussian_elution(
                w.retention_time,
                w.w[:, i],
                **fitpeaks_kwargs,
            )
            mus.append(mu)
            sigmas.append(s)
            maes.append(mae)
            keep.append(True)
        except Exception as e:
            logger.exception(f"Error fitting gaussian to component: {e}")
            mus.append(0)
            sigmas.append(0)
            maes.append(0)
            keep.append(False)
    w.set_peak_fits(np.array(mus), np.array(sigmas), np.array(maes), np.array(keep))


def fit_nmf_matrix_scanwindow(
    w: ScanWindow,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    **nmf_kwargs: Dict[str, Any],
) -> None:
    """Fit NMF decomposition to a scan window's intensity matrix.

    Args:
        w (ScanWindow): Scan window object to fit NMF to.
        W_init (Optional[np.ndarray]): Initial guess for W matrix.
        H_init (Optional[np.ndarray]): Initial guess for H matrix.
        **nmf_kwargs: Keyword arguments passed to fit_nmf_matrix_custom_init.
    """
    W, H, model = fit_nmf_matrix_custom_init(
        w.m,
        W_init=W_init,
        H_init=H_init,
        return_model=True,
        **nmf_kwargs,
    )
    w.set_nmf_fit(W, H, model)


def fit_nmf_matrix_gpfrun(
    gpfrun: GPFRun,
    n_jobs: int = 8,
    config: Config = Config(),
) -> None:
    """Fit NMF decomposition to all scan windows in a GPF run in parallel.

    Args:
        gpfrun (GPFRun): GPF run object to fit.
        n_jobs (int): Number of parallel jobs to run.
        nmf_kwargs (Dict[str, Any]): Keyword arguments for NMF fitting.
        fitpeaks_kwargs (Dict[str, Any]): Keyword arguments for peak fitting.
    """

    def modify_fit_scanwindow(sw: ScanWindow) -> ScanWindow:
        """Helper function to process a single scan window with NMF and peak fitting.

        Args:
            sw (ScanWindow): Scan window to process

        Returns:
            ScanWindow: Processed scan window with NMF decomposition and peak fits
        """
        if not sw.is_filtered:
            sw.filter_scans()
        fit_nmf_matrix_scanwindow(sw, **config.nmf)
        fit_scanwindow(sw, **config.fitpeaks)
        return sw

    with Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs") as parallel:
        fit_scan_windows = parallel(
            delayed(modify_fit_scanwindow)(sw)
            for sw in tqdm(
                gpfrun.scan_windows,
                disable=(not config.tqdm_enabled),
                total=len(gpfrun.scan_windows),
            )
        )

    gpfrun.scan_windows = fit_scan_windows
    gpfrun.collapse_redundant_components()


def fit_nmf_matrix_msrun(
    msrun: MSRun,
    n_jobs: int = 8,
    config: Config = Config(),
) -> None:
    """Fit NMF decomposition to all GPF runs in an MS run.

    Args:
        msrun (MSRun): MS run object to process
        n_jobs (int): Number of parallel jobs to run
        config (Config): Configuration object containing NMF parameters
    """
    start_time = time.time()
    logging.info("Begin fitting GPFRuns...")

    n_gpfs = len(msrun._gpf_runs)

    for gpf_index in tqdm(msrun._gpf_runs, disable=(not config.tqdm_enabled)):
        gpf_start_time = time.time()
        gpf = msrun.get_gpf(gpf_index)
        logging.info(
            f"Starting to work on GPF index: {gpf_index}/{n_gpfs} -> {gpf._gpf}."
        )
        fit_nmf_matrix_gpfrun(
            gpf,
            n_jobs=n_jobs,
            config=config,
        )
        logging.info(
            f"Finished working on GPF index: {gpf_index}/{n_gpfs} -> {gpf._gpf} in {time.time() - gpf_start_time}."
        )
    logging.info(f"Finished fitting GPFRuns in {time.time() - start_time}.")


def add_ms1_features_scanwindow(
    w: ScanWindow,
    ms1_df: pd.DataFrame,
    isotope_mu: float = Constants.isotope_mu,
    config: Config = Config(),
) -> None:
    """Add MS1 features to a scan window from a Biosaur2 features DataFrame.

    Args:
        w (ScanWindow): Scan window to add features to
        ms1_df (pd.DataFrame): DataFrame containing MS1 features from Biosaur2
        isotope_mu (float): Mass difference between isotopes
        config (Config): Configuration object
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sub_ms1_df = ms1_df.loc[
            (ms1_df["rtStart"] < w.retention_time[-1])
            & (ms1_df["rtEnd"] > w.retention_time[0])
        ]
        # If statement ensure that apply returns a series regardless of whether
        # sub_ms1_df is empty or not. Need to return series for map below.
        mz_isotopes = sub_ms1_df.apply(
            lambda x: (
                [
                    x.mz + (i * isotope_mu / x.charge)
                    for i in range(0, x.nIsotopes)
                    if x.mz + (i * isotope_mu / x.charge)
                    >= getattr(w, "_gpf_low", w._gpf - 1.0)
                    and x.mz + (i * isotope_mu / x.charge)
                    <= getattr(w, "_gpf_high", w._gpf + 1.0)
                ]
                if not sub_ms1_df.empty
                else None
            ),
            axis=1,
        )
        sub_ms1_df = sub_ms1_df.loc[mz_isotopes.map(lambda x: len(x) > 0)]
        # TODO: Replace with below:
        # mz_isotopes = sub_ms1_df.apply(
        #     lambda x: np.where(
        #         (x.mz + (np.arange(x.nIsotopes) * isotope_mu / x.charge) >= getattr(w, "_gpf_low", sw._gpf - 1.0)) &
        #         (x.mz + (np.arange(x.nIsotopes) * isotope_mu / x.charge) <= getattr(w, "_gpf_high", sw._gpf + 1.0)),
        #     )[0].sum() if not sub_ms1_df.empty else [],
        #     axis=1
        # )
        # sub_ms1_df = sub_ms1_df.loc[mz_isotopes > 0]

        ms1_features = []

        for i, r in sub_ms1_df.iterrows():
            ms1_feature = MS1Feature.from_biosaur2_series_with_ms2(
                r.copy(),
                w.retention_time.copy(),
            )
            ms1_features.append(ms1_feature)

        w.set_ms1_features(ms1_features)


def add_ms1_features_gpfrun(
    gpfrun: GPFRun,
    ms1_df: pd.DataFrame,
    config: Config = Config(),
    n_jobs: int = 8,
) -> None:
    """Add MS1 features to all scan windows in a GPF run in parallel.

    Args:
        gpfrun (GPFRun): GPF run object to process
        ms1_df (pd.DataFrame): DataFrame containing MS1 features from Biosaur2
        config (Config): Configuration object
        n_jobs (int): Number of parallel jobs to run
    """

    def modify_add_ms1_scanwindow(w: ScanWindow) -> ScanWindow:
        """Helper function to add MS1 features to a single scan window.

        Args:
            w (ScanWindow): Scan window to process

        Returns:
            ScanWindow: Scan window with added MS1 features
        """
        add_ms1_features_scanwindow(w, ms1_df, config=config)
        return w

    with Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs") as parallel:
        modified_scan_windows = parallel(
            delayed(modify_add_ms1_scanwindow)(sw)
            for sw in tqdm(
                gpfrun.scan_windows,
                disable=(not config.tqdm_enabled),
                total=len(gpfrun.scan_windows),
            )
        )

    gpfrun.scan_windows = modified_scan_windows


def add_ms1_features_msrun(
    msrun: MSRun,
    ms1_df: pd.DataFrame,
    config: Config = Config(),
    n_jobs: int = 8,
) -> None:
    """Add MS1 features to all GPF runs in an MS run.

    Args:
        msrun (MSRun): MS run object to process
        ms1_df (pd.DataFrame): DataFrame containing MS1 features from Biosaur2
        config (Config): Configuration object
        n_jobs (int): Number of parallel jobs to run
    """
    start_time = time.time()
    logging.info("Begin adding MS1 features GPFRuns...")

    n_gpfs = len(msrun._gpf_runs)

    for gpf_index in tqdm(msrun._gpf_runs, disable=(not config.tqdm_enabled)):
        gpf_start_time = time.time()
        gpf = msrun.get_gpf(gpf_index)
        logging.info(
            f"Starting to work on GPF index: {gpf_index}/{n_gpfs} -> {gpf._gpf}."
        )
        add_ms1_features_gpfrun(gpf, ms1_df, config=config, n_jobs=n_jobs)
        logging.info(
            f"Finished working on GPF index: {gpf_index}/{n_gpfs} -> {gpf._gpf} in {time.time() - gpf_start_time}."
        )

    logging.info(
        f"Finished adding MS1 Features for GPFRuns in {time.time() - start_time}."
    )


def fit_ms1_ms2_feature_matching_scanwindow(
    w: ScanWindow,
    **feature_matching_kwargs: Dict[str, Any],
) -> None:
    """Match MS1 features to MS2 components within a scan window.

    Args:
        w (ScanWindow): Scan window object containing MS1 and MS2 data
        **feature_matching_kwargs: Keyword arguments for feature matching including:
            alpha (float): L1 regularization parameter
            extend_w_fraction (float): Fraction to extend component width
            c_cutoff (float): Minimum coefficient threshold
    """
    alpha = feature_matching_kwargs.get("alpha")
    extend_w_fraction = feature_matching_kwargs.get("extend_w_fraction")
    c_cutoff = feature_matching_kwargs.get("c_cutoff")

    sw_fit_mean, sw_fit_sigma = w.component_fit_parameters
    ms1_elution_matrix, ms1_feature_name = w.ms1_features_information
    rt = w.retention_time

    ms1_ms2_feature_match = np.zeros(
        (
            w.component_indices.shape[0],
            ms1_feature_name.shape[0],
        )
    )
    global_ms2_explained_variance = np.zeros((w.component_indices.shape[0]))
    individual_ms2_explained_variance = np.zeros(
        (
            w.component_indices.shape[0],
            ms1_feature_name.shape[0],
        )
    )

    for i, s in enumerate(w.component_indices):
        # Pull 99% range of normal distribution centered around fit parameters.
        w_ci = stats.norm.interval(0.99, loc=sw_fit_mean[i], scale=sw_fit_sigma[i])
        # Extend this range by extend_w_fraction.
        w_low = w_ci[0] - (extend_w_fraction * (w_ci[1] - w_ci[0]))
        w_high = w_ci[1] + (extend_w_fraction * (w_ci[1] - w_ci[0]))
        idx_low = max(np.searchsorted(rt, w_low, side="left") - 1, 0)
        idx_high = min(np.searchsorted(rt, w_high, side="right"), rt.shape[0] - 1)

        sub_ms2 = w.w[idx_low:idx_high, i].flatten()
        ms2_norm = np.linalg.norm(sub_ms2)
        if np.isclose(ms2_norm, 0.0):
            logger.warning(
                f"Component {i} of scanwindow: {w.component_names[i]}:{w.scan_index[0]}-{w.scan_index[-1]} has zero norm."
            )
            ms1_ms2_feature_match[i, :] = 0.0
            continue
        sub_ms2 = sub_ms2 / ms2_norm

        sub_ms1_keep = ms1_elution_matrix[idx_low:idx_high]
        ms1_keep_mask = sub_ms1_keep.sum(axis=0) > 0
        sub_ms1_keep = sub_ms1_keep[:, ms1_keep_mask] / np.linalg.norm(
            ms1_elution_matrix[:, ms1_keep_mask], axis=0
        )

        if sub_ms1_keep.size == 0:
            ms1_ms2_feature_match[i, :] = 0.0
            continue

        coef = least_squares_with_l1_bounds(sub_ms1_keep, sub_ms2, alpha=alpha)
        keep_coef = ~np.isclose(coef, 0.0) & (coef > c_cutoff)
        coef[~keep_coef] = 0.0

        ms1_ms2_feature_match[i, ms1_keep_mask] = coef

        global_ms2_explained_variance[i] = fraction_explained_variance(
            sub_ms1_keep, sub_ms2, coef
        )
        individual_ms2_explained_variance[i, ms1_keep_mask] = np.array(
            [
                fraction_explained_variance(sub_ms1_keep[:, c], sub_ms2, coef[c])
                for c in range(coef.shape[0])
            ]
        )

    w.set_ms1_ms2_feature_matches(
        ms1_ms2_feature_match,
        global_ms2_explained_variance,
        individual_ms2_explained_variance,
    )


def fit_ms1_ms2_feature_matching_gpfrun(
    gpfrun: GPFRun,
    n_jobs: int = 8,
    config: Config = Config(),
) -> None:
    """Match MS1 features to MS2 components across all scan windows in a GPF run.

    Args:
        gpfrun (GPFRun): GPF run object containing scan windows
        n_jobs (int): Number of parallel jobs to run
        config (Config): Configuration object containing feature matching parameters
    """

    def modify_match_scanwindow(sw: ScanWindow) -> ScanWindow:
        """Helper function to match MS1 and MS2 features in a single scan window.

        Args:
            sw (ScanWindow): Scan window to process

        Returns:
            ScanWindow: Scan window with matched MS1 and MS2 features
        """
        fit_ms1_ms2_feature_matching_scanwindow(sw, **config.feature_matching)
        return sw

    with Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs") as parallel:
        fit_scan_windows = parallel(
            delayed(modify_match_scanwindow)(sw)
            for sw in tqdm(
                gpfrun.scan_windows,
                disable=(not config.tqdm_enabled),
                total=len(gpfrun.scan_windows),
            )
        )

    gpfrun.scan_windows = fit_scan_windows


def fit_ms1_ms2_feature_matching_msrun(
    msrun: MSRun,
    n_jobs: int = 8,
    config: Config = Config(),
) -> None:
    """Match MS1 features to MS2 components across all GPF runs in an MS run.

    Args:
        msrun (MSRun): MS run object to process
        n_jobs (int): Number of parallel jobs to run
        config (Config): Configuration object containing feature matching parameters
    """
    start_time = time.time()
    logging.info("Begin matching MS1/MS2 features GPFRuns...")

    n_gpfs = len(msrun._gpf_runs)

    for gpf_index in tqdm(msrun._gpf_runs, disable=(not config.tqdm_enabled)):
        gpf_start_time = time.time()
        gpf = msrun.get_gpf(gpf_index)
        logging.info(
            f"Starting to work on GPF index: {gpf_index}/{n_gpfs} -> {gpf._gpf}."
        )
        fit_ms1_ms2_feature_matching_gpfrun(
            gpf,
            n_jobs=n_jobs,
            config=config,
        )
        logging.info(
            f"Finished working on GPF index: {gpf_index}/{n_gpfs} -> {gpf._gpf} in {time.time() - gpf_start_time}."
        )

    logging.info(f"Finished matching GPFRuns in {time.time() - start_time}.")


def dump_features_to_df_scanwindow(
    sw: ScanWindow,
    gpfrun: GPFRun,
    config: Config = Config(),
) -> pd.DataFrame:
    """Convert scan window features to a pandas DataFrame.

    Args:
        sw (ScanWindow): Scan window containing MS1 and MS2 features
        gpfrun (GPFRun): Parent GPF run containing the scan window
        config (Config): Configuration object with feature output parameters

    Returns:
        pd.DataFrame: DataFrame containing MS2 component information and optional MS1 matches
        with columns for component properties, peak information, and matching coefficients
    """
    if config.feature_matching.match_ms1 and not sw._is_ms1_features_fit:
        raise AttributeError("MS1 features have not been matched to MS2 components")

    matches = []

    component_means, component_sigmas = sw.component_fit_parameters
    scan_number = sw.scan_number
    scan_index = sw.scan_index
    retention_time = sw.retention_time
    reverse_max_scaling = sw.reverse_transform_maxscale_scans()
    mz = sw.mz
    ms1_features = sw.ms1_features

    for i in range(sw.component_names.size):
        # Find index of closest scan to the apex.
        apex_scan_idx = np.abs(retention_time - component_means[i]).argmin()
        component_weight = sw.h[i]
        component_weight_rescaled = component_weight * reverse_max_scaling

        # Pull the observed ms2 spectra from the gpfrun object.
        sub_m_long = gpfrun.m_long[gpfrun.m_long[:, 0] == scan_number[apex_scan_idx]]

        # Find the top N peaks
        top_reweighted_peak_indices = np.argsort(component_weight_rescaled)[
            -1 * config.feature_output.top_n_rescaled_peaks :
        ][::-1]
        top_weight_indices = np.argsort(component_weight)[
            -1 * config.feature_output.top_n_rescaled_peaks :
        ][::-1]

        top_reweight_intensity = component_weight_rescaled[top_reweighted_peak_indices]
        top_reweight_mz = mz[top_reweighted_peak_indices]
        top_weight_intensity = component_weight[top_weight_indices]
        top_weight_mz = mz[top_weight_indices]

        # Build series based on ms2 feature.
        feature = {
            "ms2_component": sw.component_names[i],
            "ms2_component_rt": component_means[i],
            "ms2_component_sigma": component_sigmas[i],
            "ms2_gpf_center": sw._gpf,
            "ms2_gpf_low": sw._gpf_low,
            "ms2_gpf_high": sw._gpf_high,
            "ms2_apex_scan_index": scan_index[apex_scan_idx],
            "ms2_apex_scan_number": scan_number[apex_scan_idx],
            "ms2_apex_scan_rt": retention_time[apex_scan_idx],
            "ms2_component_weight_mz": top_weight_mz.copy(),
            "ms2_component_weight_intensity": top_weight_intensity.copy(),
            "ms2_component_reweight_mz": top_reweight_mz.copy(),
            "ms2_component_reweight_intensity": top_reweight_intensity.copy(),
            "ms2_mz_array": sub_m_long[:, 2].copy(),
            "ms2_intensity_array": sub_m_long[:, 3].copy(),
        }

        # Matching to discrete ms1 features.
        if config.feature_matching.match_ms1:
            # Add ms1 matching features, then explode.
            matching_coefficients = sw._component_ms1_coef_matrix[i]
            sorted_coef_idx = np.argsort(matching_coefficients)[::-1]
            if sorted_coef_idx.size < config.feature_output.top_n_ms1_matches:
                threshold = 0.0
            else:
                threshold = matching_coefficients[
                    sorted_coef_idx[config.feature_output.top_n_ms1_matches - 1]
                ]
            keep_coef_idx = np.where(
                (matching_coefficients >= threshold)
                & (
                    matching_coefficients
                    >= config.feature_output.matching_coefficient_threshold
                )
            )[0]

            ms1_feature_masses = np.array(
                [ms1_features[c]._calibrated_mass for c in keep_coef_idx]
            )
            ms1_feature_charges = np.array(
                [ms1_features[c]._charge for c in keep_coef_idx]
            )
            ms1_feature_mzes = np.array([ms1_features[c]._mz for c in keep_coef_idx])
            ms1_feature_apex_rt = np.array(
                [ms1_features[c]._retention_time_apex for c in keep_coef_idx]
            )
            ms1_feature_apex_intensity = np.array(
                [ms1_features[c]._intensity_apex for c in keep_coef_idx]
            )
            ms1_feature_sum_intensity = np.array(
                [ms1_features[c]._intensity_sum for c in keep_coef_idx]
            )
            ms1_feature_id = np.array([ms1_features[c]._id for c in keep_coef_idx])

            ms1_feature = {
                "ms1_matching_coef": matching_coefficients[keep_coef_idx].copy(),
                "ms1_global_explained_variance": sw._component_ms1_global_exp_var[
                    i
                ].copy(),
                "ms1_individual_explained_variance": sw._component_ms1_individual_exp_var[
                    i, keep_coef_idx
                ].copy(),
                "ms1_mass": ms1_feature_masses.copy(),
                "ms1_charge": ms1_feature_charges.copy(),
                "ms1_mz": ms1_feature_mzes.copy(),
                "ms1_apex_rt": ms1_feature_apex_rt.copy(),
                "ms1_apex_intensity": ms1_feature_apex_intensity.copy(),
                "ms1_sum_intensity": ms1_feature_sum_intensity.copy(),
                "ms1_id": ms1_feature_id.copy(),
            }
            feature.update(ms1_feature)

        matches.append(feature)

    if len(matches) == 0:
        return pd.DataFrame()

    if config.feature_matching.match_ms1:
        return (
            pd.DataFrame(matches)
            .explode(
                [
                    "ms1_matching_coef",
                    "ms1_individual_explained_variance",
                    "ms1_mass",
                    "ms1_charge",
                    "ms1_mz",
                    "ms1_apex_rt",
                    "ms1_apex_intensity",
                    "ms1_sum_intensity",
                    "ms1_id",
                ]
            )
            .reset_index(drop=True)
        )
    return pd.DataFrame(matches)


def dump_features_to_df_gpfrun(
    gpfrun: GPFRun,
    config: Config = Config(),
) -> pd.DataFrame:
    """Convert all features from a GPF run to a pandas DataFrame.

    Args:
        gpfrun (GPFRun): GPF run containing scan windows to convert
        config (Config): Configuration object with feature output parameters

    Returns:
        pd.DataFrame: Concatenated DataFrame containing features from all scan windows
        in the GPF run
    """
    dfs = []

    for sw in gpfrun.scan_windows:
        dfs.append(
            dump_features_to_df_scanwindow(
                sw,
                gpfrun,
                config=config,
            )
        )

    return pd.concat(dfs).reset_index(drop=True)


def dump_features_to_df_msrun(
    msrun: MSRun,
    config: Config = Config(),
) -> pd.DataFrame:
    """Convert all features from an MS run to a pandas DataFrame.

    Args:
        msrun (MSRun): MS run containing GPF runs to convert
        config (Config): Configuration object with feature output parameters

    Returns:
        pd.DataFrame: Concatenated DataFrame containing features from all GPF runs
        in the MS run
    """
    start_time = time.time()
    logging.info("Begin dumping GPFRuns features to dataframe...")

    dfs = []

    n_gpfs = len(msrun._gpf_runs)

    for gpf_index in tqdm(msrun._gpf_runs, disable=(not config.tqdm_enabled)):

        gpf_start_time = time.time()
        gpf = msrun.get_gpf(gpf_index)
        logging.info(
            f"Starting to work on GPF index: {gpf_index}/{n_gpfs} -> {gpf._gpf}."
        )

        dfs.append(
            dump_features_to_df_gpfrun(
                gpf,
                config=config,
            )
        )
        logging.info(
            f"Finished working on GPF index: {gpf_index}/{n_gpfs} -> {gpf._gpf} in {time.time() - gpf_start_time}."
        )

    logging.info(f"Finished dumping GPFRuns in {time.time() - start_time}.")

    return pd.concat(dfs).reset_index(drop=True)
