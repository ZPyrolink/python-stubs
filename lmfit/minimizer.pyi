from typing import Callable, Literal, NamedTuple

import emcee
import numpy as np
import numpy.typing as npt
import pandas as pd
from lmfit.parameter import Parameters

from lmfit._types import _FitMethod, _FloatBehavior, _MinimizeMethod, _NanPolicy, _PoolLike, _ReduceFcn

HAS_EMCEE: bool
HAS_PANDAS: bool
HAS_NUMDIFFTOOLS: bool


class Candidate(NamedTuple):
    params: Parameters
    score: float


maxeval_warning: str


def thisfuncname() -> str: ...


class MinimizerException(Exception):
    msg: str

    def __init__(self, msg: str) -> None: ...


class AbortFitException(MinimizerException): ...


SCALAR_METHODS: dict[str, str]


def reduce_chisquare(r: np.ndarray) -> float: ...


def reduce_negentropy(r: np.ndarray) -> float: ...


def reduce_cauchylogpdf(r: np.ndarray) -> float: ...


class MinimizerResult:
    residual: np.ndarray
    params: Parameters
    uvars: dict
    var_names: list
    covar: np.ndarray | None
    init_vals: list
    init_values: list
    aborted: bool
    status: int
    success: bool
    errorbars: bool
    message: str
    ier: int
    lmdif_message: str
    call_kws: dict
    nfev: int
    nvarys: int
    ndata: int
    chisqr: float
    redchi: float
    aic: float
    bic: float

    def __init__(self, **kws) -> None: ...

    @property
    def flatchain(self) -> pd.DataFrame: ...

    def show_candidates(self, candidate_nmb: int | Literal["all"] = 'all') -> None: ...


class Minimizer:
    userfcn: Callable
    userargs: tuple
    userkws: dict
    kws: dict
    iter_cb: Callable
    calc_covar: bool
    scale_covar: bool
    max_nfev: int | None
    nfev: int
    nfree: int
    ndata: int
    ier: int
    success: bool
    errorbars: bool
    message: None
    lmdif_message: None
    chisqr: float
    redchi: float
    covar: None
    residual: float
    reduce_fcn: _ReduceFcn
    params: Parameters
    col_deriv: bool
    jacfcn: None
    nan_policy: _NanPolicy

    def __init__(
            self,
            userfcn: Callable,
            params: Parameters,
            fcn_args: tuple | None = None,
            fcn_kws: dict | None = None,
            iter_cb: Callable | None = None,
            scale_covar: bool = True,
            nan_policy: _NanPolicy = 'raise',
            reduce_fcn: _ReduceFcn | None = None,
            calc_covar: bool = True,
            max_nfev: int | None = None,
            **kws
    ) -> None: ...

    def set_max_nfev(
            self,
            max_nfev: int | None = None,
            default_value: float = 100000
    ) -> None: ...

    @property
    def values(self): ...

    def penalty(self, fvars: np.ndarray) -> dict: ...

    result: MinimizerResult

    def prepare_fit(self, params: Parameters | None = None) -> MinimizerResult: ...

    def unprepare_fit(self) -> None: ...

    def scalar_minimize(
            self,
            method: _FitMethod = 'Nelder-Mead',
            params: Parameters | None = None,
            max_nfev: int | None = None,
            **kws
    ) -> MinimizerResult: ...

    nvarys: int
    sampler: emcee.EnsembleSampler

    def emcee(
            self,
            params: Parameters | None = None,
            steps: int = 1000,
            nwalkers: int = 100,
            burn: int = 0,
            thin: int = 1,
            ntemps: int = 1,
            pos: np.ndarray | None = None,
            reuse_sampler: bool = False,
            workers: _PoolLike | int = 1,
            float_behavior: _FloatBehavior = 'posterior',
            is_weighted: bool = True,
            seed: int | np.random.RandomState | None = None,
            progress: bool = True,
            run_mcmc_kwargs: dict = {}
    ) -> MinimizerResult: ...

    def least_squares(
            self,
            params: Parameters | None = None,
            max_nfev: int | None = None,
            **kws
    ) -> MinimizerResult: ...

    def leastsq(
            self,
            params: Parameters | None = None,
            max_nfev: int | None = None,
            **kws
    ) -> MinimizerResult: ...

    def basinhopping(
            self,
            params: Parameters | None = None,
            max_nfev: int | None = None,
            **kws
    ) -> MinimizerResult: ...

    def brute(
            self,
            params: Parameters | None = None,
            Ns: int = 20,
            keep: int = 50,
            workers: _PoolLike | int = 1,
            max_nfev: int | None = None
    ) -> MinimizerResult: ...

    def ampgo(
            self,
            params: Parameters | None = None,
            max_nfev: int | None = None,
            **kws
    ) -> MinimizerResult: ...

    def shgo(
            self,
            params: Parameters | None = None,
            max_nfev: int | None = None,
            **kws
    ) -> MinimizerResult: ...

    def dual_annealing(
            self,
            params: Parameters | None = None,
            max_nfev: int | None = None,
            **kws
    ) -> MinimizerResult: ...

    def minimize(
            self,
            method: _MinimizeMethod = 'leastsq',
            params: Parameters | None = None,
            **kws
    ) -> MinimizerResult: ...


def coerce_float64(
        arr: npt.ArrayLike,
        nan_policy: _NanPolicy = 'raise',
        handle_inf: bool = True,
        ravel: bool = True,
        ravel_order: str = 'C'
) -> npt.NDArray[np.float64]: ...


def minimize(
        fcn: Callable,
        params: Parameters,
        method: _MinimizeMethod = 'leastsq',
        args: tuple | None = None,
        kws: dict | None = None,
        iter_cb: Callable | None = None,
        scale_covar: bool = True,
        nan_policy: _NanPolicy = 'raise',
        reduce_fcn: _ReduceFcn | None = None,
        calc_covar: bool = True,
        max_nfev: int | None = None,
        **fit_kws
) -> MinimizerResult: ...
