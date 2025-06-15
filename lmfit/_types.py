from typing import Callable, Iterable, Literal, Protocol

import numpy as np

type _NanPolicy = Literal["raise", "propagate", "omit"]

type _ReduceFcn = Literal["negentropy", "neglogcauchy"] | Callable

type _FitMethod = Literal[
    "Nelder-Mead",
    "L-BFGS-B",
    "Powell",
    "CG",
    "Newton-CG",
    "COBYLA",
    "BFGS",
    "TNC",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
    "trust-constr",
    "dogleg",
    "SLSQP",
    "differential_evolution",
]

type _FloatBehavior = Literal["posterior", "chi2"]

type _MinimizeMethod = Literal[
    "leastsq",
    "least_squares",
    "differential_evolution",
    "brute",
    "basinhopping",
    "ampgo",
    "nelder",
    "lbfgsb",
    "powell",
    "cg",
    "newton",
    "cobyla",
    "bfgs",
    "tnc",
    "trust-ncg",
    "trust-exact",
    "trust-krylov",
    "trust-constr",
    "dogleg",
    "slsqp",
    "emcee",
    "shgo",
    "dual_annealing"
]

type _Reducer = Literal["real", "imag", "abs", "angle"]

type _EvalResult = np.ndarray | float | int | complex

type _EvalOp = Callable[[_EvalResult, _EvalResult], _EvalResult]

type _CorrelMode = Literal["list", "table"]

type _TermalDistribution = Literal["bose", "maxwell", "fermi"]

type _StepForm = Literal["linear", "atan", "erf", "logistic"]

type _Fmt = Literal["g", "e", "f"]


class _PoolLike[_S, _T](Protocol):
    def map(
            self,
            func: Callable[[_S], _T],
            iterable: Iterable[_S],
            chunksize: int | None = None
    ) -> list[_T]: ...


class _ArrayLike(Protocol):
    def __array__(self): ...


class _Writable(Protocol):
    def write(self, content: str, /) -> int: ...


class _Readable(Protocol):
    def read(self, size: int | None = ..., /) -> str: ...
