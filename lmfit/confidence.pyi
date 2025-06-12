from typing import Any, Callable, Final, Literal, overload

import numpy as np

from .minimizer import Minimizer, MinimizerResult
from .parameter import Parameter, Parameters

CONF_ERR_GEN: Final[str]
CONF_ERR_STDERR: Final[str]
CONF_ERR_NVARS: Final[str]


def f_compare(best_fit: MinimizerResult, new_fit: MinimizerResult) -> float: ...


def copy_vals(params: Parameters) -> dict[str, Any]: ...


def restore_vals(tmp_params: dict[str, Any], params: Parameters) -> None: ...


@overload
def conf_interval(
        minimizer: Minimizer,
        result: MinimizerResult,
        p_names: list[str] | None = None,
        sigmas: list[int] | None = None,
        trace: Literal[True] = ...,
        maxiter: int = 200,
        verbose: bool = False,
        prob_func: Callable | None = None,
        min_rel_change: float = 1e-05
) -> dict[str, dict[str, list]]: ...


@overload
def conf_interval(
        minimizer: Minimizer,
        result: MinimizerResult,
        p_names: list[str] | None = None,
        sigmas: list[int] | None = None,
        trace: Literal[False] = ...,
        maxiter: int = 200,
        verbose: bool = False,
        prob_func: Callable | None = None,
        min_rel_change: float = 1e-05
) -> dict[str, tuple[int, int]]: ...


def map_trace_to_names(trace: dict, params: Parameters) -> dict: ...


class ConfidenceInterval:
    verbose: bool
    minimizer: Minimizer
    result: MinimizerResult
    params: Parameters
    org: dict[str, Any]
    best_chi: float
    p_names: list[str]
    fit_params: list[int]
    prob_func: Callable
    trace_dict: dict[str, list]
    trace: bool
    maxiter: int
    min_rel_change: float
    sigmas: list[int]
    probs: list[int]

    def __init__(
            self,
            minimizer: Minimizer,
            result: MinimizerResult,
            p_names: list[str] | None = None,
            prob_func: Callable | None = None,
            sigmas: list[int] | None = None,
            trace: bool = False,
            verbose: bool = False,
            maxiter: int = 50,
            min_rel_change: float = 1e-05
    ) -> None: ...

    def calc_all_ci(self) -> dict[str, float]: ...

    def calc_ci(self, para: Parameter, direction: float) -> list[tuple[float, float]]: ...

    def reset_vals(self) -> None: ...

    def find_limit(self, para: Parameter, direction: float) -> tuple[float, float]: ...

    def calc_prob(
            self,
            para: Parameter,
            val: float,
            offset: float = 0.0,
            restore: bool = False
    ) -> float: ...


def conf_interval2d(
        minimizer: Minimizer,
        result: MinimizerResult,
        x_name: str,
        y_name: str,
        nx: int = 10,
        ny: int = 10,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
        prob_func: None = None,  # Deprecated
        nsigma: int = 5,
        chi2_out: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
