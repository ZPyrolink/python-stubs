import _ast
from typing import Callable, ClassVar, Final, Literal, LiteralString

import numpy as np
from asteval import asteval
from lmfit.model import Model as Model, ModelResult
from lmfit.parameter import Parameters

from lmfit._types import _EvalResult, _MinimizeMethod, _NanPolicy, _StepForm, _TermalDistribution

tau: float


class DimensionalError(Exception): ...


def fwhm_expr(model: Model) -> LiteralString: ...


def height_expr(model: Model) -> LiteralString: ...


def guess_from_peak(
        model: Model,
        y: np.ndarray,
        x: np.ndarray,
        negative: bool,
        ampscale: float = 1.0,
        sigscale: float = 1.0
) -> Parameters: ...


def guess_from_peak2d(
        model: Model,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        negative: bool
) -> Parameters: ...


def update_param_vals(
        pars: Parameters,
        prefix: str,
        **kwargs
) -> Parameters: ...


COMMON_INIT_DOC: str
COMMON_GUESS_DOC: str


def _default__init__(
        self,
        independent_vars: list[str] = ['x'],
        prefix: str = '',
        nan_policy: _NanPolicy = 'raise',
        **kwargs
) -> None: ...


def _default_eval(
        self,
        params: Parameters | None = None,
        *,
        x: np.ndarray,
        **kwargs,
) -> _EvalResult: ...


def _default_fit(
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
        *,
        x: np.ndarray,
        **kwargs
) -> ModelResult: ...


class _DefaultModel(Model):
    __init__ = _default__init__
    # type: ignore[bad-override]
    eval = _default_eval
    # type: ignore[bad-override]
    fit = _default_fit


class _NegativeModel(_DefaultModel):
    guess = _negative_guess


class _FormModel[_Lit](Model):
    valid_forms: ClassVar[Final[tuple[str, ...]]]

    def __init__(
            self,
            independent_vars: list[str] = ['x', 'form'],
            prefix: str = '',
            nan_policy: _NanPolicy = 'raise',
            form: _Lit = ...,
            **kwargs
    ) -> None: ...


def _negative_guess(
        self,
        data: np.ndarray,
        x: np.ndarray,
        negative: bool = False,
        **kwargs
) -> Parameters: ...


class ConstantModel(_DefaultModel): ...


class ComplexConstantModel(_DefaultModel): ...


class LinearModel(_DefaultModel): ...


class QuadraticModel(_DefaultModel): ...


ParabolicModel = QuadraticModel


class PolynomialModel(_DefaultModel):
    MAX_DEGREE: ClassVar[Final[Literal[7]]]
    DEGREE_ERR: ClassVar[Final[Final[LiteralString]]]
    valid_forms: ClassVar[Final[tuple[int, ...]]]

    poly_degree: int

    def __init__(
            self,
            degree: int = 7,
            independent_vars: list[str] = ['x'],
            prefix: str = '',
            nan_policy: _NanPolicy = 'raise',
            **kwargs
    ) -> None: ...


class SplineModel(_DefaultModel):
    MAX_KNOTS: ClassVar[Final[int]]
    NKNOTS_MAX_ERR: ClassVar[Final[str]]
    NKNOTS_NDARRY_ERR: ClassVar[Final[str]]
    DIM_ERR: ClassVar[Final[str]]
    xknots: list | tuple | np.ndarray
    nknots: int
    order: int

    def __init__(
            self,
            xknots: list | tuple | np.ndarray,
            independent_vars: list[str] = ['x'],
            prefix: str = '',
            nan_policy: _NanPolicy = 'raise',
            **kwargs
    ) -> None: ...


class SineModel(_DefaultModel): ...


class _FwhmModel(_DefaultModel):
    fwhm_factor: ClassVar[Final[float]]


class _HeightModel(_DefaultModel):
    height_factor: ClassVar[Final[float]]


class _GaussianModel(_FwhmModel, _HeightModel): ...


class GaussianModel(_GaussianModel): ...


class Gaussian2dModel(Model):
    def __init__(
            self,
            independent_vars: list[str] = ['x', 'y'],
            prefix: str = '',
            nan_policy: _NanPolicy = 'raise',
            **kwargs
    ) -> None: ...

    # type: ignore[bad-override]
    def guess(
            self,
            data: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            negative: bool = False,
            **kwargs
    ) -> Parameters: ...

    # type: ignore[bad-override]
    def eval(
            self,
            params: Parameters | None = None,
            *,
            x: np.ndarray,
            y: np.ndarray,
            **kwargs
    ) -> _EvalResult: ...

    # type: ignore[bad-override]
    def fit(
            self,
            data: np.ndarray,
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
            *,
            x: np.ndarray,
            y: np.ndarray,
            **kwargs
    ) -> ModelResult: ...


class LorentzianModel(_GaussianModel): ...


class SplitLorentzianModel(_NegativeModel): ...


class VoigtModel(Model): ...


class PseudoVoigtModel(_FwhmModel): ...


class MoffatModel(_NegativeModel): ...


class Pearson4Model(_NegativeModel): ...


class Pearson7Model(_NegativeModel): ...


class StudentsTModel(_NegativeModel): ...


class BreitWignerModel(_NegativeModel): ...


class LognormalModel(_NegativeModel): ...


class DampedOscillatorModel(_HeightModel): ...


class DampedHarmonicOscillatorModel(_FwhmModel): ...


class ExponentialGaussianModel(_NegativeModel): ...


class SkewedGaussianModel(_NegativeModel): ...


class SkewedVoigtModel(_NegativeModel): ...


class ThermalDistributionModel(_FormModel[_TermalDistribution]):
    guess = _negative_guess


class DoniachModel(_NegativeModel): ...


class PowerLawModel(_DefaultModel): ...


class ExponentialModel(_DefaultModel): ...


class StepModel(_FormModel[_StepForm]): ...


class RectangleModel(_FormModel[_StepForm]): ...


class ExpressionModel(Model):
    idvar_missing: ClassVar[Final[str]]
    idvar_notfound: ClassVar[Final[str]]
    no_prefix: ClassVar[Final[str]]
    asteval: asteval.Interpreter
    expr: str
    astcode: _ast.Module
    independent_var_defvals: dict
    def_vals: dict

    def __init__(
            self,
            expr: str,
            independent_vars: list[str] | None = None,
            init_script: str | None = None,
            nan_policy: _NanPolicy = 'raise',
            **kws
    ) -> None: ...


lmfit_models: dict[str, type[Model]]
