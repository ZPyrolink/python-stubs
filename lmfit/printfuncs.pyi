import re

from lmfit.minimizer import MinimizerResult
from lmfit.model import ModelResult
from lmfit.parameter import Parameters

from lmfit._types import _CorrelMode

HAS_NUMDIFFTOOLS: bool


def alphanumeric_sort(s: str, _nsre: re.Pattern = ...) -> list[int]: ...


def getfloat_attr(obj: object, attr: str, length: int = 11) -> str: ...


def gformat(val: float, length: int = 11) -> str: ...


def fit_report(
        inpars: Parameters,
        modelpars: Parameters | None = None,
        show_correl: bool = True,
        min_correl: float = 0.1,
        sort_pars: bool = False,
        correl_mode: _CorrelMode = 'list'
) -> str: ...


def lcol(s: str, cat: str = 'td') -> str: ...


def rcol(s: str, cat: str = 'td') -> str: ...


def trow(columns: list[str], cat: str = 'td') -> list[str]: ...


def fitreport_html_table(
        result: MinimizerResult | ModelResult,
        show_correl: bool = True,
        min_correl: float = 0.1
) -> str: ...


def correl_table(params: Parameters) -> str: ...


def params_html_table(params: Parameters) -> str: ...


def report_fit(params: Parameters, **kws) -> None: ...


def ci_report(
        ci: dict,
        with_offset: bool = True,
        ndigits: int = 5
) -> str: ...


def report_ci(ci: dict) -> None: ...
