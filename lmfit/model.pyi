from typing import Any, Callable, NoReturn

import matplotlib.pyplot as plt
import numpy as np

from ._types import _ArrayLike, _CorrelMode, _EvalOp, _EvalResult, _MinimizeMethod, _NanPolicy, _Readable, _Reducer, \
    _Writable
from .minimizer import Minimizer
from .parameter import Parameters

tiny: float


def get_reducer(option: _Reducer) -> Callable: ...


def propagate_err(
        z: np.ndarray,
        dz: np.ndarray,
        option: _Reducer
) -> np.ndarray: ...


def coerce_arraylike(x: _ArrayLike): ...


class Model:
    func: Callable
    independent_vars: list[str]
    nan_policy: _NanPolicy
    opts: dict[str, Any]
    independent_vars_defvals: list[str]
    param_hints: dict[str, Any]

    def __init__(
            self,
            func: Callable,
            independent_vars: list[str] | None = None,
            param_names: list[str] | None = None,
            nan_policy: _NanPolicy = 'raise',
            prefix: str = '',
            name: str | None = None,
            **kws
    ) -> None: ...

    def dumps(self, **kws) -> str: ...

    def dump(self, fp: _Writable, **kws) -> int: ...

    def loads(self, s: str, funcdefs: dict[str, Callable] | None = None, **kws) -> Model: ...

    def load(self, fp: _Writable, funcdefs: dict[str, Callable] | None = None, **kws) -> Model: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, value: str) -> None: ...

    @property
    def prefix(self) -> str: ...

    @prefix.setter
    def prefix(self, value: str) -> None: ...

    @property
    def param_names(self) -> list[str]: ...

    def copy(self, **kwargs) -> NoReturn: ...

    def set_param_hint(self, name: str, **kwargs) -> None: ...

    def print_param_hints(self, colwidth: int = 8) -> None: ...

    def make_params(self, verbose: bool = False, **kwargs) -> Parameters: ...

    def guess(self, data: np.ndarray, x: np.ndarray, **kwargs) -> Parameters: ...

    def make_funcargs(
            self,
            params: Parameters | None = None,
            kwargs: dict | None = None,
            strip: bool = True
    ) -> dict: ...

    def post_fit(self, fitresult: ModelResult) -> None: ...

    def eval(
            self,
            params: Parameters | None = None,
            **kwargs
    ) -> _EvalResult: ...

    @property
    def components(self) -> list[Model]: ...

    def eval_components(
            self,
            params: Parameters | None = None,
            **kwargs
    ) -> dict: ...

    def fit(
            self,
            data,
            params: Parameters | None = None,
            weights: np.ndarray | None = None,
            method: _MinimizeMethod = 'leastsq',
            iter_cb: Callable | None = None,
            scale_covar: bool = True,
            verbose: bool = False,
            fit_kws: dict | None = None,
            nan_policy: _NanPolicy | None = None,
            calc_covar: bool = True,
            max_nfev: int | None = None,
            coerce_farray: bool = True,
            **kwargs
    ) -> ModelResult: ...

    def __add__(self, other) -> CompositeModel: ...

    def __sub__(self, other) -> CompositeModel: ...

    def __mul__(self, other) -> CompositeModel: ...

    def __truediv__(self, other) -> CompositeModel: ...


class CompositeModel(Model):
    left: Model
    right: Model
    op: _EvalOp

    def __init__(
            self,
            left: Model,
            right: Model,
            op: _EvalOp,
            **kws
    ) -> None: ...

    def eval_components(self, **kwargs) -> dict: ...


def save_model(model: Model, fname: str) -> None: ...


def load_model(fname: str, funcdefs: dict | None = None) -> Model: ...


def save_modelresult(modelresult: ModelResult, fname: str) -> None: ...


def load_modelresult(fname: str, funcdefs: dict | None = None): ...


class ModelResult(Minimizer):
    model: Model
    data: np.ndarray | None
    weights: np.ndarray | None
    method: _MinimizeMethod
    ci_out: dict[str, dict[str, list]] | dict[str, tuple[int, int]] | None
    user_options: None
    init_params: Parameters

    def __init__(
            self,
            model: Model,
            params: Parameters,
            data: np.ndarray | None = None,
            weights: np.ndarray | None = None,
            method: _MinimizeMethod = 'leastsq',
            fcn_args: list[str] | None = None,
            fcn_kws: dict | None = None,
            iter_cb: Callable | None = None,
            scale_covar: bool = True,
            nan_policy: _NanPolicy = 'raise',
            calc_covar: bool = True,
            max_nfev: int | None = None,
            **fit_kws
    ) -> None: ...

    init_fit: _EvalResult
    init_values: dict[str, Any]
    best_values: dict[str, Any]
    best_fit: _EvalResult
    rsquared: float

    def fit(
            self,
            data: np.ndarray | None = None,
            params: Parameters | None = None,
            weights: np.ndarray | None = None,
            method: _MinimizeMethod | None = None,
            nan_policy: _NanPolicy | None = None,
            **kwargs
    ) -> None: ...

    eval = Model.eval

    eval_components = Model.eval_components

    dely: float
    dely_predicted: float
    dely_comps: dict

    def eval_uncertainty(
            self,
            params: Parameters | None = None,
            sigma: float = 1,
            dscale: float = 0.01,
            **kwargs
    ) -> np.ndarray: ...

    # ToDo: overload
    def conf_interval(self, **kwargs) -> dict[str, dict[str, list]]: ...

    def ci_report(
            self,
            with_offset: bool = True,
            ndigits: int = 5,
            **kwargs
    ) -> str: ...

    def fit_report(
            self,
            modelpars: Parameters | None = None,
            show_correl: bool = True,
            min_correl: float = 0.1,
            sort_pars: bool = False,
            correl_mode: _CorrelMode = 'list'
    ) -> str: ...

    def summary(self) -> dict: ...

    def dumps(self, **kws) -> str: ...

    def dump(self, fp: _Writable, **kws) -> int: ...

    uvars: dict
    init_vals: list[float]

    def loads(
            self,
            s: str,
            funcdefs: dict | None = None,
            **kws
    ) -> ModelResult: ...

    def load(
            self,
            fp: _Readable,
            funcdefs: dict | None = None,
            **kws
    ) -> ModelResult: ...

    def plot_fit(
            self,
            ax: plt.Axes | None = None,
            datafmt: str = 'o',
            fitfmt: str = '-',
            initfmt: str = '--',
            xlabel: str | None = None,
            ylabel: str | None = None,
            yerr: np.ndarray | None = None,
            numpoints: int | None = None,
            data_kws: dict | None = None,
            fit_kws: dict | None = None,
            init_kws: dict | None = None,
            ax_kws: dict | None = None,
            show_init: bool = False,
            parse_complex: _Reducer = 'abs',
            title: str | None = None
    ) -> plt.Axes: ...

    def plot_residuals(
            self,
            ax: plt.Axes | None = None,
            datafmt: str = 'o',
            yerr: np.ndarray[] | None = None,
            data_kws: dict | None = None,
            fit_kws: dict | None = None,
            ax_kws: dict | None = None,
            parse_complex: str = 'abs',
            title: str | None = None
    ) -> plt.Axes: ...

    def plot(
            self,
            datafmt: str = 'o',
            fitfmt: str = '-',
            initfmt: str = '--',
            xlabel: str | None = None,
            ylabel: str | None = None,
            yerr: np.ndarray | None = None,
            numpoints: int | None = None,
            fig: plt.Figure | None = None,
            data_kws: dict | None = None,
            fit_kws: dict | None = None,
            init_kws: dict | None = None,
            ax_res_kws: dict | None = None,
            ax_fit_kws: dict | None = None,
            fig_kws: dict | None = None,
            show_init: bool = False,
            parse_complex: _Reducer = 'abs',
            title: str | None = None
    ) -> plt.Figure: ...
