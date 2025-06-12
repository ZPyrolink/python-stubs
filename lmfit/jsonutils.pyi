from typing import Any, Callable, overload

import numpy as np
import uncertainties
from pandas import DataFrame, Series

HAS_DILL: bool
pyvers: str


def find_importer(obj: Any) -> str | None: ...


def import_from(modulepath: str, objectname: str) -> Any: ...


@overload
def encode4js(
        obj: (DataFrame |
              Series |
              uncertainties.core.AffineScalarFunc |
              np.ndarray |
              complex |
              tuple |
              list |
              Callable)
) -> dict: ...


@overload
def encode4js[_T](obj: _T) -> _T: ...


def decode4js(obj: dict) -> Any: ...
