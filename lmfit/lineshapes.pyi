from typing import Literal, SupportsAbs, TypeVar

import numpy as np
import numpy.typing as npt

log2: np.float64
s2pi: np.float64
s2: np.float64
tiny: float
functions: tuple[str, ...]

_T = TypeVar("_T")


def not_zero(value: SupportsAbs) -> float: ...


def gaussian(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0
) -> npt.NDArray[_T]: ...


def gaussian2d(
        x: npt.NDArray[_T],
        y: float = 0.0,
        amplitude: float = 1.0,
        centerx: float = 0.0,
        centery: float = 0.0,
        sigmax: float = 1.0,
        sigmay: float = 1.0
) -> npt.NDArray[_T]: ...


def lorentzian(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0
) -> npt.NDArray[_T]: ...


def split_lorentzian(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        sigma_r: float = 1.0
) -> npt.NDArray[_T]: ...


def voigt(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        gamma: float | None = None
) -> npt.NDArray[_T]: ...


def pvoigt(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        fraction: float = 0.5
) -> npt.NDArray[_T]: ...


def moffat(
        x: npt.NDArray[_T],
        amplitude: int = 1,
        center: float = 0.0,
        sigma: int = 1,
        beta: float = 1.0
) -> npt.NDArray[_T]: ...


def pearson4(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        expon: float = 1.0,
        skew: float = 0.0
) -> npt.NDArray[_T]: ...


def pearson7(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        expon: float = 1.0
) -> npt.NDArray[_T]: ...


def breit_wigner(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        q: float = 1.0
) -> npt.NDArray[_T]: ...


def damped_oscillator(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 1.0,
        sigma: float = 0.1
) -> npt.NDArray[_T]: ...


def dho(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 1.0,
        sigma: float = 1.0,
        gamma: float = 1.0
) -> npt.NDArray[_T]: ...


def logistic(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0
) -> npt.NDArray[_T]: ...


def lognormal(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: int = 1
) -> npt.NDArray[_T]: ...


def students_t(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0
) -> npt.NDArray[_T]: ...


def expgaussian(
        x: npt.NDArray[_T],
        amplitude: int = 1,
        center: int = 0,
        sigma: float = 1.0,
        gamma: float = 1.0
) -> npt.NDArray[_T]: ...


def doniach(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: int = 0,
        sigma: float = 1.0,
        gamma: float = 0.0
) -> npt.NDArray[_T]: ...


def skewed_gaussian(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        gamma: float = 0.0
) -> npt.NDArray[_T]: ...


def skewed_voigt(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        gamma: float | None = None,
        skew: float = 0.0
) -> npt.NDArray[_T]: ...


def sine(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        frequency: float = 1.0,
        shift: float = 0.0
) -> npt.NDArray[_T]: ...


def expsine(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        frequency: float = 1.0,
        shift: float = 0.0,
        decay: float = 0.0
) -> npt.NDArray[_T]: ...


def thermal_distribution(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        kt: float = 1.0,
        form: Literal["bose", "maxwell", "fermi"] = 'bose'
) -> npt.NDArray[_T]: ...


def step(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center: float = 0.0,
        sigma: float = 1.0,
        form: str = 'linear'
) -> npt.NDArray[_T]: ...


def rectangle(
        x: npt.NDArray[_T],
        amplitude: float = 1.0,
        center1: float = 0.0,
        sigma1: float = 1.0,
        center2: float = 1.0,
        sigma2: float = 1.0,
        form: Literal["linear", "atan", "arctan", "erf", "logisitic"] = 'linear'
) -> npt.NDArray[_T]: ...


def exponential(
        x: npt.NDArray[_T],
        amplitude: int = 1,
        decay: int = 1
) -> npt.NDArray[_T]: ...


def powerlaw(
        x: npt.NDArray[_T],
        amplitude: int = 1,
        exponent: float = 1.0
) -> npt.NDArray[_T]: ...


def linear(
        x: npt.NDArray[_T],
        slope: float = 1.0,
        intercept: float = 0.0
) -> npt.NDArray[_T]: ...


def parabolic(
        x: npt.NDArray[_T],
        a: float = 0.0,
        b: float = 0.0,
        c: float = 0.0
) -> npt.NDArray[_T]: ...
