"""Collects how closely each case actually agreed, not just whether it passed.

A case sitting just inside tolerance and one sitting eight orders of
magnitude inside it both show up as a green tick, and they are not the same
news. The suite prints the worst relative deviation per case at the end of
the run so that a port drifting towards its tolerance is visible before it
crosses.
"""

from __future__ import annotations

#: case id -> (worst relative deviation, which field it was on).
AGREEMENT: dict[str, tuple[float, str]] = {}


def record_agreement(case_id: str, deviation: float, field: str) -> None:
    """Record one case's worst relative deviation from the reference.

    Parameters
    ----------
    case_id : str
        Case directory name.
    deviation : float
        Worst relative deviation over all fields and wedges.
    field : str
        Which field and wedge that was.
    """
    AGREEMENT[case_id] = (deviation, field)
