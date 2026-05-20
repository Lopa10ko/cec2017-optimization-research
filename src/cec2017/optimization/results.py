from dataclasses import dataclass


@dataclass(frozen=True)
class TrialResult:
    f_min: float
    elapsed_sec: float
