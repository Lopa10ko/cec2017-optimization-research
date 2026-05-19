from collections.abc import Callable

import cec2017.functions as cec_functions

Function = Callable


class CEC2017Benchmark:
    """Access to CEC-2017 single-objective benchmark functions.

    Wraps the upstream ``cec2017-py`` implementation. Functions expect input
    arrays of shape ``(m, D)`` and return a 1D array of length ``m``.

    Attributes:
        DOMAIN: Inclusive bounds for each dimension, per CEC-2017 definition.
    """

    DOMAIN: tuple[float, float] = (-100.0, 100.0)

    def get_function(self, name: str) -> Function:
        return getattr(cec_functions, name)

    def unimodal_functions(self) -> list[tuple[str, Function]]:
        return [(f"f{i}", self.get_function(f"f{i}")) for i in range(1, 11)]

    def functions_range(self, start: int, end: int) -> list[tuple[str, Function]]:
        return [(f"f{i}", self.get_function(f"f{i}")) for i in range(start, end + 1)]
