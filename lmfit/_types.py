from typing import Callable, Iterable, Literal, Protocol

type _NanPolicy = Literal[
    "raise",
    "propagate",
    "omit"
]

type _ReduceFcn = Literal[
                      "negentropy",
                      "neglogcauchy"
                  ] | Callable

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


class _PoolLike[_S, _T](Protocol):
    def map(
            self,
            func: Callable[[_S], _T],
            iterable: Iterable[_S],
            chunksize: int | None = None
    ) -> list[_T]: ...
