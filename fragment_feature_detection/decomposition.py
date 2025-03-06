import base64
import logging
import numbers
import pickle
import platform
import tempfile
import time
import warnings
from collections import defaultdict
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager, Queue
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import NMF
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.utils._param_validation import Interval
from sklearn.utils.parallel import Parallel, delayed

from fragment_feature_detection.config import Config, format_logger
from fragment_feature_detection.utils import calculate_nmf_summary

logger = logging.getLogger(__name__)

_FIT_AND_SCORE = [
    "_test_reconstruction_errors",
    "_weight_orthogonality",
    "_sample_orthogonality",
    "_nonzero_component_fraction",
    "_neglog_ratio_train_test_reconstruction_error",
    "_mean_weight_sparsity",
    "_mean_sample_sparsity",
    "_fraction_window_component",
]


class OptimizationParameters:
    """Parameters for optimizing NMF decomposition.

    This class handles optimization parameters and scoring for NMF decomposition,
    including component counts, scan widths, and error metrics.

    Attributes:
        _components_in_window (float): Target number of components per window
        _components (float): Total number of components
        _scan_width (float): Width of scan window
        _component_sigma (float): Component sigma value
        _error (str): Error metric type ('l1' or 'l2')
        _scores (dict): Target scores for different optimization metrics
    """

    _components_in_window = 8.0
    _components = 20.0
    _scan_width = 150.0
    _component_sigma = 3.0
    _error = "l1"

    def __init__(self, error: Literal["l1", "l2"] = "l1"):
        """ """
        self._error = error
        self.set_params()

    def set_params(self) -> None:
        """Sets target scores for different optimization metrics.

        Updates the internal _scores dictionary with target values for various
        optimization metrics based on the current parameter settings.
        """
        self._scores = {
            "weight_orthogonality": 0.0,
            "sample_orthogonality": 0.0,
            "nonzero_component_fraction": self._components_in_window / self._components,
            "mean_weight_sparsity": -1.0,
            "mean_sample_sparsity": -1.0,
            "fraction_window_component": (
                self._components_in_window * 4 * self._component_sigma
            )
            / self._scan_width,
            # 'fraction_window_component': 0.0,
        }

    def score(
        self, param: str, value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Calculates optimization score for a parameter value.

        Args:
            param (str): Name of parameter to score
            value (Union[float, np.ndarray]): Parameter value(s) to score

        Returns:
            Union[float, np.ndarray]: Calculated score based on error metric
        """
        if param in self._scores.keys():
            if self._error == "l1":
                return -1.0 * np.abs(value + self._scores[param])
            elif self._error == "l2":
                return -1.0 * (value + self._scores[param]) ** 2
        return value

    @classmethod
    def from_config(cls, config: Config = Config()) -> "OptimizationParameters":
        """Creates OptimizationParameters instance from config object.

        Args:
            config (Config): Configuration object containing optimization parameters

        Returns:
            OptimizationParameters: New instance initialized with config values
        """
        obj = cls()

        obj._components_in_window = config.tuning.components_in_window
        obj._components = config.tuning.n_components
        obj._scan_width = config.scan_filter.scan_width
        # component_sigma is scale paramter of normal distribution of component in units of scans, not in RT
        obj._component_sigma = config.tuning.component_sigma

        obj.set_params()

        return obj


class MzBinMaskingSplitter:
    """Custom splitter that randomly masks bins in scan windows for cross-validation.

    This splitter creates train/test splits by masking random mass-to-charge (m/z) bins
    in scan windows, allowing evaluation of NMF reconstruction performance.

    Args:
        n_splits (int): Number of train/test splits to generate
        mask_fraction (float): Fraction of bins to mask in each split
        random_state (int): Random seed for reproducibility
        mask_signal (bool): Whether to only mask bins containing signal
        balance_mask_signal (bool): Whether to balance masked signal vs non-signal bins

    Attributes:
        _n_splits (int): Number of splits
        _mask_fraction (float): Fraction to mask
        _random_state (int): Random seed
        _balance_mask_signal (bool): Balance masking flag
        _mask_signal (bool): Signal masking flag
    """

    def __init__(
        self,
        n_splits: int = 5,
        mask_fraction: float = 0.2,
        random_state: int = 42,
        mask_signal: bool = False,
        balance_mask_signal: bool = True,
    ):
        self._n_splits = n_splits
        self._mask_fraction = mask_fraction
        self._random_state = random_state
        self._balance_mask_signal = balance_mask_signal
        self._mask_signal = mask_signal

    def split(self, X: List[np.ndarray], y=None):
        """Generate train-test masks by randomly hiding mass bins in scanwindows"""
        rng = np.random.default_rng(seed=self._random_state)
        for _ in range(self._n_splits):
            unmasked = []
            masked = []
            for m in X:
                if not self._mask_signal:
                    sub_mask = rng.random(m.shape) < self._mask_fraction
                else:
                    flat_mask = np.zeros_like(m.flatten(), dtype=bool)
                    non_zero_idxes = np.where((m > 0).flatten())[0]
                    mask_non_zero_idxes = (
                        rng.random(non_zero_idxes.shape) < self._mask_fraction
                    )
                    flat_mask[non_zero_idxes[mask_non_zero_idxes]] = True
                    if self._balance_mask_signal:
                        zero_idxes = np.where((m == 0).flatten())[0]
                        mask_zero_idxes = rng.random(zero_idxes.shape) < (
                            flat_mask.sum() / flat_mask.size
                        )
                        flat_mask[zero_idxes[mask_zero_idxes]] = True
                    sub_mask = flat_mask.reshape(m.shape)
                unmasked.append(~sub_mask)
                masked.append(sub_mask)
            yield unmasked, masked

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._n_splits

    @staticmethod
    def mask_train_matrix(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Masks values in training matrix.

        Args:
            m (np.ndarray): Input matrix to mask
            mask (np.ndarray): Boolean mask array

        Returns:
            np.ndarray: Masked matrix with zeros in masked positions
        """
        m[mask] = 0.0
        return m

    @staticmethod
    def reconstruction_error(
        m: np.ndarray,
        m_reconstructed: np.ndarray,
        mask: np.ndarray,
        *args: Any,
    ) -> float:
        """Calculates reconstruction error on masked values.

        Args:
            m (np.ndarray): Original matrix
            m_reconstructed (np.ndarray): Reconstructed matrix
            mask (np.ndarray): Boolean mask array
            *args: Additional arguments (unused)

        Returns:
            float: Mean squared error between original and reconstructed masked values
        """
        return np.nanmean((m[mask] - m_reconstructed[mask]) ** 2)


class MzBinSampleSplitter:
    """Custom splitter that samples scans in scan windows for cross-validation.

    This splitter creates train/test splits by randomly sampling entire scans
    from scan windows, allowing evaluation of NMF reconstruction performance.

    Args:
        n_splits (int): Number of train/test splits to generate
        mask_fraction (float): Fraction of scans to mask in each split
        random_state (int): Random seed for reproducibility
        **kwargs: Additional keyword arguments

    Attributes:
        _n_splits (int): Number of splits
        _mask_fraction (float): Fraction to mask
        _random_state (int): Random seed
    """

    def __init__(
        self,
        n_splits: int = 5,
        mask_fraction: float = 0.2,
        random_state: int = 42,
        **kwargs: Dict[str, Any],
    ):
        self._n_splits = n_splits
        self._mask_fraction = mask_fraction
        self._random_state = random_state

    def split(self, X: List[np.ndarray], y=None):
        """Generate train-test masks by randomly hiding mass bins in scanwindows"""
        rng = np.random.default_rng(seed=self._random_state)
        for _ in range(self._n_splits):
            unmasked = []
            masked = []
            for m in X:
                sub_mask = rng.random(m.shape[0]) < self._mask_fraction
                unmasked.append(~sub_mask)
                masked.append(sub_mask)
            yield unmasked, masked

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._n_splits

    @staticmethod
    def mask_train_matrix(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """ """
        return m[~mask]

    @staticmethod
    def reconstruction_error(
        m: np.ndarray,
        m_reconstructed: np.ndarray,
        mask: np.ndarray,
        model: NMF,
    ) -> float:
        """ """
        if model.components_.sum() == 0.0 or m[mask].sum() == 0.0:
            return np.nanmean((m[mask] - np.zeros_like(m[mask])) ** 2)
        mt = model.transform(m[mask])
        m_reconstructed = mt @ model.components_

        return np.nanmean((m[mask] - m_reconstructed) ** 2)


class NMFMaskWrapper(BaseEstimator):
    """Wrapper for NMF that handles masked cross-validation.

    This class wraps NMF models to handle masked cross-validation splits,
    calculating various performance metrics across splits.

    Args:
        splitter_type (str): Type of splitter to use ('Sample' or 'Masking')
        n_splits (int): Number of cross-validation splits
        mask_fraction (float): Fraction of data to mask
        mask_signal (bool): Whether to only mask signal-containing bins
        balance_mask_signal (bool): Whether to balance masked signal bins
        **nmf_kwargs: Additional keyword arguments for NMF

    Attributes:
        splitter_type (str): Type of splitter
        mask_fraction (float): Masking fraction
        n_splits (int): Number of splits
        mask_signal (bool): Signal masking flag
        balance_mask_signal (bool): Balance masking flag
        _nmf_kwargs (dict): NMF parameters
        _models (list): Fitted NMF models
        _reconstruction_errors (list): Reconstruction error scores
        _test_reconstruction_errors (list): Test set reconstruction errors
        _train_reconstruction_errors (list): Training set reconstruction errors
        _neglog_ratio_train_test_reconstruction_error (list): Log ratios of train/test errors
        _nonzero_component_fraction (list): Fraction of nonzero components
        _fraction_window_component (list): Component fractions per window
        _weight_orthogonality (list): Weight matrix orthogonality scores
        _sample_orthogonality (list): Sample matrix orthogonality scores
        _mean_weight_sparsity (list): Mean weight matrix sparsity
        _mean_sample_sparsity (list): Mean sample matrix sparsity
    """

    def __init__(
        self,
        splitter_type: Literal["Sample", "Masking"] = "Masking",
        n_splits: int = 5,
        mask_fraction: float = 0.2,
        mask_signal: bool = True,
        balance_mask_signal: bool = True,
        **nmf_kwargs: Dict[str, Any],
    ):
        self.splitter_type = splitter_type
        self._rng = np.random.default_rng(seed=nmf_kwargs.get("random_state", 42))
        self.mask_fraction = mask_fraction
        self.n_splits = n_splits
        self.mask_signal = mask_signal
        self.balance_mask_signal = balance_mask_signal
        self._nmf_kwargs = nmf_kwargs

    def fit_model(self, m: np.ndarray):
        """Fits NMF model to input matrix.

        Args:
            m (np.ndarray): Input matrix to fit

        Returns:
            Tuple[np.ndarray, np.ndarray, NMF]: Weight matrix, component matrix, and fitted model
        """
        W, H, model = fit_nmf_matrix_custom_init(
            m,
            return_model=True,
            **self._nmf_kwargs,
        )
        return W, H, model

    def fit(self, X, y=None):
        """Fits the NMF model with masked cross-validation.

        Performs multiple fits with different random masks to evaluate reconstruction
        performance and calculate various metrics.

        Args:
            X: Input data matrices
            y: Ignored, present for sklearn compatibility

        Returns:
            self: Returns self for method chaining
        """
        self._models = []
        self._reconstruction_errors = []
        self._test_reconstruction_errors = []
        self._train_reconstruction_errors = []
        self._neglog_ratio_train_test_reconstruction_error = []
        self._nonzero_component_fraction = []
        self._fraction_window_component = []
        self._weight_orthogonality = []
        self._sample_orthogonality = []
        self._test_samples = []
        self._train_samples = []
        self._mean_weight_sparsity = []
        self._mean_sample_sparsity = []

        if self.splitter_type == "Mask":
            splitter = MzBinMaskingSplitter(
                n_splits=self.n_splits,
                mask_fraction=self.mask_fraction,
                mask_signal=self.mask_signal,
                balance_mask_signal=self.balance_mask_signal,
                random_state=self._nmf_kwargs.get("random_state", 42),
            )
        elif self.splitter_type == "Sample":
            splitter = MzBinSampleSplitter(
                n_splits=self.n_splits,
                mask_fraction=self.mask_fraction,
                random_state=self._nmf_kwargs.get("random_state", 42),
            )
        else:
            raise ValueError(
                f"Invalid splitter type: '{self.splitter_type}'. Must be either 'Masking' or 'Sample'."
            )

        for train_idxes, test_idxes in splitter.split(X):
            for m, train_mask, test_mask in zip(X, train_idxes, test_idxes):
                m_train = m.copy()
                m_train = splitter.mask_train_matrix(m_train, test_mask)
                W, H, model = self.fit_model(m_train)

                summary_results = calculate_nmf_summary(W, H)

                m_reconstructed = W @ H

                self._models.append(model)

                mse = splitter.reconstruction_error(
                    m,
                    m_reconstructed,
                    test_mask,
                    model,
                )
                mse_train = splitter.reconstruction_error(
                    m,
                    m_reconstructed,
                    train_mask,
                    model,
                )

                self._nonzero_component_fraction.append(
                    summary_results["nonzero_component_fraction"]
                )
                self._mean_weight_sparsity.append(summary_results["weight_sparsity"])
                self._mean_sample_sparsity.append(summary_results["sample_sparsity"])
                self._weight_orthogonality.append(
                    summary_results["weight_deviation_identity"]
                )
                self._sample_orthogonality.append(
                    summary_results["sample_deviation_identity"]
                )
                self._neglog_ratio_train_test_reconstruction_error.append(
                    -1.0 * np.log2(mse_train / mse)
                )
                self._test_reconstruction_errors.append(mse)
                self._train_reconstruction_errors.append(mse_train)
                self._test_samples.append(test_mask.sum())
                self._train_samples.append(train_mask.sum())
                self._fraction_window_component.append(
                    summary_results["fraction_window_component"]
                )

        return self

    def score(self, X, y=None):
        """Calculates overall score as negative mean reconstruction error.

        Args:
            X: Input data (unused)
            y: Ignored, present for sklearn compatibility

        Returns:
            float: Negative mean reconstruction error score
        """
        return -1 * np.nanmean(self._reconstruction_errors)

    def get_params(self, deep: bool = True):
        """Gets parameters for this estimator.

        Override get_params to allow access to nmf hyperparameters of interest.

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators

        Returns:
            dict: Parameter names mapped to their values
        """
        return {**super().get_params(deep=deep), **self._nmf_kwargs}

    def set_params(self, **params: Dict[str, Any]):
        """Sets the parameters of this estimator.

        Override set_params to update the parameters in _nmf_kwargs.

        Args:
            **params: Estimator parameters

        Returns:
            self: Estimator instance
        """
        for param, value in params.items():
            if param in self._nmf_kwargs.keys():
                self._nmf_kwargs[param] = value
            else:
                setattr(self, param, value)
        return self


class RandomizedSearchReconstructionCV(BaseSearchCV):
    """Cross-validation with randomized parameter search for NMF reconstruction.

    Performs randomized search over parameter spaces for NMF models, evaluating
    reconstruction performance through cross-validation.

    Args:
        estimator: NMF estimator to optimize
        param_distributions: Dictionary with parameters names (string) as keys and
            distributions or lists of parameters to try
        n_iter (int): Number of parameter settings sampled
        scoring: Strategy to evaluate predictions on the test set
        n_jobs (int): Number of jobs to run in parallel
        verbose (int): Verbosity level
        pre_dispatch (str): Controls the number of jobs that get dispatched
        random_state (int): Random seed for reproducibility
        error_score (float): Value to assign to the score if an error occurs
        return_train_score (bool): Whether to return training scores

    Attributes:
        param_distributions: Parameter distributions for search
        n_iter (int): Number of iterations
        random_state (int): Random seed
        best_index_ (int): Index of best model
        best_score_ (float): Score of best model
        best_params_ (dict): Parameters of best model
        cv_results (dict): Cross-validation results
        n_splits_ (int): Number of splits used
    """

    _required_parameters = ["estimator", "param_distributions"]

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates: Callable):
        """ """
        evaluate_candidates(
            ParameterSampler(
                self.param_distributions,
                self.n_iter,
                random_state=self.random_state,
            )
        )

    def fit(
        self, X: Union[List[Any], np.ndarray], y=None, **params: Dict[str, Any]
    ) -> "RandomizedSearchReconstructionCV":
        """ """
        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        n_splits = self.estimator.n_splits

        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def objective(
                estimator: BaseEstimator,
                X: Union[List[Any], np.ndarray],
                log_queue: Queue,
                iter_n: int,
                extra_scores: Optional[List[str]] = None,
                parameters: Dict[str, Any] = {},
                **kwargs: Dict[str, Any],
            ) -> List[Dict]:
                """ """
                iter_logger = logging.getLogger(f"iter_{iter_n}")
                iter_logger.addHandler(QueueHandler(log_queue))
                format_logger(iter_logger, level=logging.INFO)
                iter_logger.info(f"Iter {iter_n} starting with params: {parameters}")

                return _fit_and_score(
                    estimator,
                    X,
                    parameters=parameters,
                    extra_scores=extra_scores,
                    **kwargs,
                )

            def evaluate_candidates(
                candidate_params: ParameterSampler,
                more_results: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                """ """
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                with Manager() as manager:
                    # Create a queue for logging
                    log_queue = manager.Queue()
                    # Get the root logger and create a queue listener
                    logger = logging.getLogger()
                    listener = QueueListener(log_queue, *logger.handlers)
                    listener.start()

                    out = parallel(
                        delayed(objective)(
                            clone(base_estimator),
                            X,
                            log_queue,
                            i,
                            parameters=parameters,
                            extra_scores=_FIT_AND_SCORE,
                        )
                        for i, parameters in enumerate(candidate_params)
                    )

                    listener.stop()

                out = [i for j in out for i in j]

                if len(out) < 1:
                    raise ValueError("No fits were performed.")
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "Returned fits not consistent with size of parameter space."
                    )

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results

                results = self._format_results(
                    all_candidate_params,
                    n_splits,
                    all_out,
                    all_more_results,
                )

                return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_score"].argmin()
        self.best_score_ = results["mean_test_score"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.cv_results = results
        self.n_splits_ = n_splits

        return self


class OptunaSearchReconstructionCV(BaseSearchCV):
    """Cross-validation with Optuna-based parameter search for NMF reconstruction.

    Uses Optuna for hyperparameter optimization of NMF models, supporting
    multi-objective optimization and pruning of poor parameter combinations.

    Args:
        estimator: NMF estimator to optimize
        param_sampling: Parameter space definition for Optuna
        objective_params: Parameters to optimize in objective function
        n_iter (int): Number of trials
        scoring: Strategy to evaluate predictions
        n_jobs (int): Number of parallel jobs
        verbose (int): Verbosity level
        pre_dispatch (str): Controls job dispatching
        random_state (int): Random seed
        error_score (float): Score to assign on error
        return_train_score (bool): Whether to return training scores
        prune_bad_parameter_spaces (bool): Whether to prune poor parameter combinations
        optimization_parameters (OptimizationParameters): Parameters for optimization

    Attributes:
        param_sampling: Parameter sampling strategy
        n_iter (int): Number of iterations
        random_state (int): Random seed
        objective_params: Objective parameters
        prune_bad_parameter_spaces (bool): Pruning flag
        optimization_parameters (OptimizationParameters): Optimization parameters
        best_index_ (int): Index of best model
        best_score_ (float): Score of best model
        best_params_ (dict): Parameters of best model
        best_trials_ (list): Best trials from optimization
        cv_results (dict): Cross-validation results
        n_splits_ (int): Number of splits used

    Class Attributes:
        _prune_at_zero (list): Parameters to prune when zero
        _required_parameters (list): Required initialization parameters
        _parameter_constraints (dict): Constraints on parameters
    """

    _prune_at_zero = [
        "_weight_orthogonality",
        "_sample_orthogonality",
        "_nonzero_component_fraction",
        "_mean_weight_sparsity",
        "_mean_sample_sparsity",
        "_fraction_window_component",
    ]

    _required_parameters = ["estimator", "param_distributions"]

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator,
        param_sampling,
        objective_params,
        *,
        n_iter=10,
        scoring=None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        random_state: Optional[int] = None,
        error_score: float = np.nan,
        return_train_score: bool = False,
        prune_bad_parameter_spaces: bool = False,
        optimization_parameters: OptimizationParameters = OptimizationParameters(),
    ):
        self.param_sampling = param_sampling
        self.n_iter = n_iter
        self.random_state = random_state
        self.objective_params = objective_params
        self.prune_bad_parameter_spaces = prune_bad_parameter_spaces
        self.optimization_parameters = optimization_parameters
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates: Callable):
        """ """
        evaluate_candidates(self.param_sampling)

    def fit(
        self, X: Union[List[Any], np.ndarray], y=None, **params: Dict[str, Any]
    ) -> "RandomizedSearchReconstructionCV":
        """ """
        base_estimator = clone(self.estimator)

        n_splits = self.estimator.n_splits

        results = {}
        best_trials = []

        all_more_results = defaultdict(list)

        def evaluate_candidates(
            candidate_params: Dict[str, Tuple[Any]],
            more_results: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """ """

            def objective(trial: optuna.Trial) -> Union[np.ndarray, Tuple[np.ndarray]]:
                """Objective function for Optuna optimization.

                Evaluates a trial's parameter set by fitting models and calculating scores.

                Args:
                    trial (optuna.Trial): Current optimization trial

                Returns:
                    Union[float, Tuple[float, ...]]: Score(s) for the trial

                Raises:
                    optuna.TrialPruned: If trial should be pruned
                """
                results = None
                params = {}
                try:
                    for k, v in candidate_params.items():
                        # Suggest categorical variable.
                        if isinstance(v[0], str):
                            params[k] = trial.suggest_categorical(k, v)
                        # Suggest continuous variable.
                        elif isinstance(v[0], (float, int)):
                            params[k] = trial.suggest_float(
                                k,
                                *v[:2],
                                log=True if (len(v) > 2 and v[2] == "log") else False,
                            )

                    results = _fit_and_score(
                        clone(base_estimator),
                        X,
                        parameters=params,
                        extra_scores=_FIT_AND_SCORE,
                    )

                    if any([r["fit_error"] for r in results]):
                        raise Exception("One or more fits failed in cv.")

                    trial.set_user_attr(
                        "out", base64.b64encode(pickle.dumps(results)).decode("utf-8")
                    )

                    try:
                        output_parameters = []
                        for k in self.objective_params:
                            parameter_value = np.nanmean(
                                [r["test_scores"][k] for r in results]
                            )
                            if (
                                self.prune_bad_parameter_spaces
                                and f"_{k}" in self._prune_at_zero
                                and parameter_value == 0.0
                            ):
                                parameter_value = -10.0
                            parameter_value = self.optimization_parameters.score(
                                k, parameter_value
                            )
                            output_parameters.append(parameter_value)
                        return tuple(output_parameters)
                    except Exception as e:
                        print("Only one objective defined.", e)

                    return np.nanmean([r["test_scores"] for r in results])

                except Exception as e:
                    if results is not None:
                        print([r["fit_error"] for r in results])
                    print(e)
                    trial.set_user_attr("error", str(e))
                    trial.set_user_attr(
                        "out",
                        base64.b64encode(
                            pickle.dumps([None] * int(base_estimator.n_splits))
                        ).decode("utf-8"),
                    )
                    raise optuna.TrialPruned()

            def optimize_optuna_study(
                study_name: str,
                storage_string: str,
                objective: Callable,
                n_trials: int,
                worker_n: int,
                log_queue: Queue,
            ) -> None:
                """Runs Optuna optimization study.

                Args:
                    study_name (str): Name for the study
                    storage_string (str): Storage location for study results
                    objective (Callable): Objective function to optimize
                    n_trials (int): Number of trials to run
                """
                working_logger = logging.getLogger(f"worker_{worker_n}")
                working_logger.addHandler(QueueHandler(log_queue))
                format_logger(working_logger, level=logging.INFO)
                working_logger.info(f"Worker {worker_n} starting...")

                class OptunaLoggingCallback:
                    """Callback for logging Optuna optimization progress.

                    Logs trial results, parameters, and optimization progress.
                    """

                    def __init__(self, logger: logging.Logger, worker: int) -> None:
                        """
                        Args:
                            logger (logging.Logger): Instance of worker logger.
                            worker (int): Worker number.
                        """
                        self.logger = logger
                        self.worker = worker

                    def __call__(
                        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
                    ) -> None:
                        """ """
                        if "error" in trial.user_attrs:
                            error_msg = trial.user_attrs.get("error", "Unknown error")
                            self.logger.error(
                                f"Failed trial: {trial.number} with params: {trial.params} on {self.worker}, {error_msg}."
                            )
                        else:
                            self.logger.info(
                                f"Completed trial: {trial.number} with params: {trial.params} on {self.worker}."
                            )

                study = optuna.create_study(
                    study_name=study_name, storage=storage_string, load_if_exists=True
                )
                callback = OptunaLoggingCallback(working_logger, worker_n)
                study.optimize(objective, n_trials=n_trials, callbacks=[callback])

            with tempfile.NamedTemporaryFile(
                dir=tempfile.gettempdir(), suffix=".db"
            ) as temp_db, Manager() as manager:
                # Get root logger, create queue, start a queuelistiner, and pass queue to parallel func.
                logger = logging.getLogger()
                log_queue = manager.Queue()
                listener = QueueListener(log_queue, *logger.handlers)
                listener.start()

                # TODO: Figure out why windows temp storage isn't working for optuna database.
                # Try this: storage_string = f"sqlite:///{str(Path(temp_db.name).resolve().as_posix())}"
                if platform.system() == "Windows":
                    storage_string = f"sqlite:///{str((Path().cwd() / 'optuna_tuning_db.db').resolve())}"
                else:
                    storage_string = f"sqlite:///{str(Path(temp_db.name).resolve())}"

                study = optuna.create_study(
                    storage=storage_string,
                    directions=["maximize" for _ in self.objective_params],
                )

                with Parallel(
                    n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch
                ) as parallel:
                    parallel(
                        delayed(optimize_optuna_study)(
                            study.study_name,
                            storage_string,
                            objective,
                            (self.n_iter // self.n_jobs) + 1,
                            i,
                            log_queue,
                        )
                        for i in range(self.n_jobs)
                    )

                listener.stop()

            all_out = [
                r
                for t in study.trials
                for r in pickle.loads(base64.b64decode(t.user_attrs["out"]))
                if not t.user_attrs.get("error", None)
            ]

            if len(all_out) < 1:
                raise ValueError("No fits were performed.")
            elif len(all_out) < self.n_iter * n_splits:
                raise ValueError(
                    "Returned fits not consistent with size of parameter space."
                )

            all_candidate_params = [
                t.params for t in study.trials if not t.user_attrs.get("error", None)
            ]

            if more_results:
                for key, value in more_results.items():
                    all_more_results[key].extend(value)

            nonlocal results

            results = self._format_results(
                all_candidate_params,
                n_splits,
                all_out,
                all_more_results,
            )

            nonlocal best_trials

            best_trials = [
                {
                    "params": t.params,
                    "scores": dict(zip(self.objective_params, t._values)),
                }
                for t in study.best_trials
            ]

            return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_score"].argmin()
        self.best_score_ = results["mean_test_score"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]
        self.best_trials_ = best_trials

        self.cv_results = results
        self.n_splits_ = n_splits

        return self


def _fit_and_score(
    estimator: BaseEstimator,
    X: Union[List[Any], np.ndarray],
    extra_scores: Optional[List[str]] = None,
    parameters: Dict[str, Any] = {},
    **kwargs: Dict[str, Any],
) -> List[Dict]:
    """Fits an estimator and calculates scores for cross-validation.

    Args:
        estimator (BaseEstimator): Scikit-learn estimator to fit
        X (Union[List[Any], np.ndarray]): Training data
        extra_scores (Optional[List[str]]): Additional scoring metrics to calculate
        parameters (Dict[str, Any]): Parameters to set on the estimator
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        List[Dict]: List of dictionaries containing for each fold:
            - test_scores: Test set scores
            - train_scores: Training set scores
            - n_test_samples: Number of test samples
            - estimator: Fitted estimator
            - fit_time: Time taken to fit
            - score_time: Time taken to score
            - fit_error: Any error during fitting
    """
    start_time = time.time()
    fit_error = None
    score_time = 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.set_params(**parameters)
            estimator.fit(X)
    except Exception as e:
        fit_time = time.time() - start_time
        fit_error = str(e)
        return [
            {
                "estimator": estimator._models[i],
                "fit_time": fit_time,
                "score_time": score_time,
                "fit_error": fit_error,
            }
            for i in range(estimator.n_splits)
        ]
    fit_time = time.time() - start_time

    results = [
        {
            "test_scores": -1.0 * estimator._test_reconstruction_errors[i],
            "train_scores": -1.0 * estimator._train_reconstruction_errors[i],
            "n_test_samples": estimator._test_samples[i],
            "estimator": estimator._models[i],
            "fit_time": fit_time,
            "score_time": score_time,
            "fit_error": fit_error,
        }
        for i in range(estimator.n_splits)
    ]

    if extra_scores and all(hasattr(estimator, es) for es in extra_scores):
        for i, r in enumerate(results):
            r["test_scores"] = {
                "score": r["test_scores"],
            }
            r["test_scores"].update(
                {
                    es.lstrip("_"): -1.0 * getattr(estimator, es)[i]
                    for es in extra_scores
                }
            )
            r["train_scores"] = {
                "score": r["train_scores"],
            }
            r["train_scores"].update(
                {
                    es.lstrip("_"): -1.0 * getattr(estimator, es)[i]
                    for es in extra_scores
                }
            )

    return results


def fit_nmf_matrix_custom_init(
    m: np.ndarray,
    n_components: int = 20,
    alpha_W: float = 0.00001,
    alpha_H: float = 0.0375,
    l1_ratio: float = 0.75,
    max_iter: int = 500,
    solver: str = "cd",
    random_state: int = 42,
    init: Optional[str] = None,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    return_model: bool = False,
    **nmf_kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Optional[NMF]]:
    """Fits NMF model with optional custom initialization.

    Args:
        m (np.ndarray): Input data matrix
        n_components (int): Number of components to extract
        alpha_W (float): L1/L2 regularization parameter for W matrix
        alpha_H (float): L1/L2 regularization parameter for H matrix
        l1_ratio (float): Ratio of L1 vs L2 regularization
        max_iter (int): Maximum number of iterations
        solver (str): NMF solver to use
        random_state (int): Random seed
        init (Optional[str]): Initialization method
        W_init (Optional[np.ndarray]): Initial W matrix
        H_init (Optional[np.ndarray]): Initial H matrix
        return_model (bool): Whether to return fitted model
        **nmf_kwargs (Dict[str, Any]): Additional keyword arguments for NMF

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[NMF]]:
            - W matrix (components)
            - H matrix (activations)
            - Fitted NMF model (if return_model=True)
    """
    init = (
        "custom"
        if isinstance(W_init, np.ndarray) and isinstance(H_init, np.ndarray)
        else (init if init else ("nndsvd" if solver == "cd" else "nndsvda"))
    )
    n_components = min(n_components, max(min(m.shape), 1))
    model = NMF(
        n_components=n_components,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        solver=solver,
        init=init,
        random_state=random_state,
        **nmf_kwargs,
    )
    try:
        if min(m.shape) == 0:
            # Output empty basis and weight matrix.
            W = np.zeros((m.shape[0], n_components))
            model.components_ = np.zeros((n_components, m.shape[1]))
            model.reconstruction_err_ = 1e6
        else:
            if isinstance(W_init, np.ndarray) and isinstance(H_init, np.ndarray):
                W = model.fit_transform(m, W=W_init, H=H_init)
            else:
                W = model.fit_transform(m)
    except Exception as e:
        # This exception getting raised in scipy linalg somewhere during nmf initialization.
        # str(e) == "array must not contain infs or NaNs"

        # Output empty basis and weight matrix.
        W = np.zeros((m.shape[0], n_components))
        model.components_ = np.zeros((n_components, m.shape[1]))
        model.reconstruction_err_ = 1e6

        logger.exception(e)

    if return_model:
        return W, model.components_, model
    return W, model.components_


def pick_parameters_harmonic_mean(
    bcv: BaseSearchCV,
    parameter_names: List[str],
    constant: float = 0.001,
    optimization_parameters: OptimizationParameters = OptimizationParameters(),
) -> Dict[str, Any]:
    """Selects optimal parameters using harmonic mean of cross-validation scores.

    Args:
        bcv (BaseSearchCV): Fitted cross-validation object
        parameter_names (List[str]): Names of parameters to optimize
        constant (float): Small constant to avoid division by zero
        optimization_parameters (OptimizationParameters): Parameters for optimization

    Returns:
        Dict[str, Any]: Dictionary of optimal parameters selected using harmonic mean
            of cross-validation scores

    Raises:
        ValueError: If any parameter name is not found in cv_results
    """
    if not all([f"mean_test_{p}" in bcv.cv_results.keys() for p in parameter_names]):
        raise ValueError(
            "All parameter names must be a valid parameter name in cv_results."
        )

    parameters = {p: bcv.cv_results[f"mean_test_{p}"] for p in parameter_names}

    # reset params if the target value was different than reported here.
    parameters = {k: optimization_parameters.score(k, v) for k, v in parameters.items()}

    # min-max scale all parameters.
    parameters = {p: (v - v.min()) / (v - v.min()).max() for p, v in parameters.items()}

    harmonic_mean = len(parameters) / (
        1.0
        / (
            np.vstack(
                list(parameters.values()),
            )
            + constant
        )
    ).sum(axis=0)

    return harmonic_mean


def tune_hyperparameters_randomizedsearchcv(
    ms: List[np.ndarray],
    config: Config = Config(),
    save_run: bool = True,
    n_jobs: int = 4,
) -> None:
    """Tune hyperparameters using RandomizedSearchCV.

    Args:
        ms (List[np.ndarray]): List of matrices to tune on
        config (Config): Configuration object containing tuning parameters
        save_run (bool): Whether to save the tuning results

    Returns:
        Dict[str, Any]: Best parameters found by harmonic mean scoring
    """
    # Get root logger setup
    logger = logging.getLogger()
    original_logger_state = logger.disabled
    original_logger_level = logger.level

    log_suffix = f"tuning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    base_file_name = Path("randomized_tuning.log")

    # Create new file handler
    old_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base_file_name = Path(handler.baseFilename)
            base_file_name = base_file_name.with_suffix(f".{log_suffix}.log")
            new_handler = logging.FileHandler(
                base_file_name,
                mode="w",
            )
            new_handler.setFormatter(handler.formatter)
            new_handler.setLevel(handler.level)
            logger.addHandler(new_handler)
            old_handler = handler
            logger.removeHandler(old_handler)
            break

    # Add a filehandler if one didn't already exist on root logger
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        new_handler = logging.FileHandler(
            base_file_name.with_suffix(f".{log_suffix}.log"),
            mode="w",
        )
        logger.addHandler(new_handler)

    # Format all handlers on logger.
    format_logger(logger)

    # Create parameters to optimize from config
    op = OptimizationParameters.from_config(config)

    rcv = RandomizedSearchReconstructionCV(
        NMFMaskWrapper(
            splitter_type=config.tuning.splitter_type,
            mask_fraction=config.tuning.test_fraction,
            n_components=config.tuning.n_components,
            n_splits=config.tuning.n_splits,
            l1_ratio=config.nmf.l1_ratio,
            alpha_W=config.nmf.alpha_W,
            alpha_H=config.nmf.alpha_H,
        ),
        config.tuning.hyperparameter_grid,
        n_iter=config.tuning.n_iter,
        random_state=config.random_seed,
        return_train_score=True,
        n_jobs=n_jobs,
    )

    logger.info("Starting hyperparameter optimization")
    logger.info(f"Number of iterations: {config.tuning.n_iter}")
    logger.info(f"Parameter space: {config.tuning.hyperparameter_grid}")

    rcv.fit([m.copy() for m in ms])

    logger.info("Optimization complete")
    logger.info(f"Best parameters: {rcv.best_params_}")
    logger.info(f"Best score: {rcv.best_score_}")

    # Calculate harmonic mean scores
    harmonic_scores = pick_parameters_harmonic_mean(
        rcv, config.tuning.objective_params, optimization_parameters=op
    )
    best_harmonic_idx = np.argmax(harmonic_scores)
    logger.info(
        f"Best parameters by harmonic mean: {rcv.cv_results['params'][best_harmonic_idx]}"
    )
    logger.info(f"Best harmonic mean score: {harmonic_scores[best_harmonic_idx]}")

    if save_run:
        pickle_fh = base_file_name.with_suffix(".pkl")
        with open(pickle_fh, "wb") as f:
            pickle.dump(rcv, f)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler is new_handler:
                logger.removeHandler(handler)
                handler.close()

    if old_handler is not None:
        logger.addHandler(old_handler)

    logger.disabled = original_logger_state
    logger.setLevel(original_logger_level)

    return rcv.cv_results["params"][best_harmonic_idx]


def tune_hyperparameters_optunasearchcv(
    ms: List[np.ndarray],
    config: Config = Config(),
    save_run: bool = True,
    n_jobs: int = 4,
) -> Dict[str, Any]:
    """Tune hyperparameters using OptunaSearchCV.

    Performs hyperparameter optimization using Optuna with multiple objectives.
    Logs results and optionally saves the optimization run.

    Args:
        ms (List[np.ndarray]): List of matrices to tune on
        config (Config): Configuration object containing tuning parameters
        save_run (bool): Whether to save the tuning results

    Returns:
        Dict[str, Any]: Best parameters found by harmonic mean scoring
    """
    # Get root logger setup.
    logger = logging.getLogger()
    original_logger_state = logger.disabled
    original_logger_level = logger.level

    log_suffix = f"tuning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    base_file_name = Path("optuna_tuning.log")

    # Create new file handler.
    old_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base_file_name = Path(handler.baseFilename)
            base_file_name = base_file_name.with_suffix(f".{log_suffix}.log")
            new_handler = logging.FileHandler(
                base_file_name,
                mode="w",
            )
            new_handler.setFormatter(handler.formatter)
            new_handler.setLevel(handler.level)
            logger.addHandler(new_handler)
            old_handler = handler
            logger.removeHandler(old_handler)
            break

    # Add a filehandler if one didn't already exist on root logger.
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        new_handler = logging.FileHandler(
            base_file_name.with_suffix(f".{log_suffix}.log"),
            mode="w",
        )
        logger.addHandler(new_handler)

    # Format all handlers on logger.
    format_logger(logger)

    optuna.logging.disable_default_handler()
    optuna.logging.enable_propagation()

    # Create parameters to optimize fromm config.
    op = OptimizationParameters.from_config(config)

    ocv = OptunaSearchReconstructionCV(
        NMFMaskWrapper(
            splitter_type=config.tuning.splitter_type,
            mask_fraction=config.tuning.test_fraction,
            n_components=config.tuning.n_components,
            n_splits=config.tuning.n_splits,
            l1_ratio=config.nmf.l1_ratio,
            alpha_W=config.nmf.alpha_W,
            alpha_H=config.nmf.alpha_H,
        ),
        config.tuning.optuna_hyperparameter_grid,
        objective_params=config.tuning.objective_params,
        n_iter=config.tuning.n_iter,
        random_state=config.random_seed,
        return_train_score=True,
        n_jobs=n_jobs,
        prune_bad_parameter_spaces=False,
        optimization_parameters=op,
    )

    logger.info("Starting hyperparameter optimization")
    logger.info(f"Number of iterations: {config.tuning.n_iter}")
    logger.info(f"Parameter space: {config.tuning.optuna_hyperparameter_grid}")

    ocv.fit([m.copy() for m in ms])

    logger.info("Optimization complete")

    for trial in ocv.best_trials_:
        logger.info(f"Trial parameters: {trial['params']}")
        logger.info(f"Trial scores: {trial['scores']}")

    # Calculate harmonic mean scores
    harmonic_scores = pick_parameters_harmonic_mean(
        ocv, config.tuning.objective_params, optimization_parameters=op
    )
    best_harmonic_idx = np.argmax(harmonic_scores)
    logger.info(
        f"Best parameters by harmonic mean: {ocv.cv_results['params'][best_harmonic_idx]}"
    )
    logger.info(f"Best harmonic mean score: {harmonic_scores[best_harmonic_idx]}")

    if save_run:
        pickle_fh = base_file_name.with_suffix(".pkl")
        with open(pickle_fh, "wb") as f:
            pickle.dump(ocv, f)

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler is new_handler:
                logger.removeHandler(handler)
                handler.close()

    if old_handler is not None:
        logger.addHandler(old_handler)

    # Disable propagation on the optuna logger.
    optuna.logging.disable_propagation()

    logger.disabled = original_logger_state
    logger.setLevel(original_logger_level)

    return ocv.cv_results["params"][best_harmonic_idx]
