"""Choosing a data-collection setting by gradient.

:func:`~pyraddose_3d.solve.solve` is differentiable in every setting in
:data:`~pyraddose_3d.solve.TUNABLE`, and torch supplies optimisers for
several settings at once. This module adds what that leaves out for the
commonest single question -- which energy? -- and nothing more.

Energy needs care for one reason: absorption edges. At each, the crystal's
absorption jumps, and so does anything computed from it. A gradient is
blind to a jump -- it describes the slope where it stands, and the slope on
either side of an edge can point straight at it -- so a search that crosses
one can end somewhere worse than where it started. The edges are known in
advance, though, so :func:`maximise_energy` splits the range at them and
searches each smooth piece on its own.

Within a piece, the gradient's sign says which side of a point the best
energy lies, so the search bisects on it: each run halves the interval,
with no step size to tune. That assumes one maximum per piece -- true of
everything tried so far, and worth checking with a coarse scan for a new
objective.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pyraddose_3d.backend.container import elemental_edges, mixture_edges
from pyraddose_3d.solve import crystal_material, solve

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pyraddose_3d.backend.summary import WedgeSummary
    from pyraddose_3d.models.experiment import Experiment

__all__ = [
    "Search",
    "absorption_edges",
    "diffraction_efficiency",
    "edge_free_intervals",
    "maximise_energy",
    "maximise_on_interval",
]

#: How far each piece stops short of an edge by default, in keV: the edge
#: itself belongs to one side or the other depending on the table, and a
#: search should evaluate neither.
_EDGE_CLEARANCE = 1e-6


@dataclass(frozen=True)
class Search:
    """One search over an interval: every point evaluated, in order.

    Attributes
    ----------
    low, high : float
        The interval searched.
    points : list[tuple[float, float, float]]
        ``(x, value, slope)`` for each evaluation.
    """

    low: float
    high: float
    points: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def best(self) -> tuple[float, float]:
        """The best point found, as ``(x, value)``."""
        x, value, _ = max(self.points, key=lambda point: point[1])
        return x, value

    @property
    def evaluations(self) -> int:
        """How many times the objective and its gradient were computed."""
        return len(self.points)


def diffraction_efficiency(summaries: list[WedgeSummary]) -> Any:
    """Elastically scattered photons per MGy, over every wedge together.

    RADDOSE-3D reports a diffraction efficiency per wedge: that wedge's
    elastic yield over its average diffraction-weighted dose. For the whole
    experiment this sums the yields and weights each wedge's DWD by its
    yield -- the dose the diffraction saw, on average. With one wedge it is
    that wedge's figure exactly.

    Parameters
    ----------
    summaries : list[WedgeSummary]
        What :func:`~pyraddose_3d.solve.solve` returns.

    Returns
    -------
    Any
        Photons per MGy, carrying any gradient the summaries do.
    """
    if len(summaries) == 1:
        return summaries[0].diffraction_efficiency
    elastic: Any = 0.0
    weighted_dose: Any = 0.0
    for summary in summaries:
        elastic = elastic + summary.elastic_yield
        weighted_dose = weighted_dose + summary.elastic_yield * summary.average_dwd
    return elastic * elastic / weighted_dose


def maximise_on_interval(
    objective: Callable[[Any], Any],
    low: float,
    high: float,
    *,
    tolerance: float = 0.01,
) -> Search:
    """Find the maximum of a smooth, single-peaked function on an interval.

    Bisects on the sign of the gradient: positive means the maximum lies
    above, negative below. If the gradient already points out of the
    interval at an end, the maximum is that end.

    Parameters
    ----------
    objective : Callable[[Any], Any]
        Takes a scalar float64 tensor and returns a scalar tensor that
        depends on it differentiably.
    low, high : float
        The interval.
    tolerance : float
        Stop once the maximum is bracketed this closely.

    Returns
    -------
    Search
        Every point evaluated. At most ``2 + ceil(log2((high - low) /
        tolerance))`` of them.
    """
    import torch

    search = Search(low, high)

    def slope(x: float) -> float:
        point = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        value = objective(point)
        (gradient,) = torch.autograd.grad(value, point)
        search.points.append((x, value.item(), gradient.item()))
        return gradient.item()

    if slope(low) <= 0:
        return search
    if slope(high) >= 0:
        return search
    below, above = low, high
    while above - below > tolerance:
        middle = (below + above) / 2
        if slope(middle) > 0:
            below = middle
        else:
            above = middle
    return search


def absorption_edges(experiment: Experiment, low: float, high: float) -> list[float]:
    """Energies between ``low`` and ``high`` where the crystal's absorption jumps.

    The edges of every element in the crystal, and of its container if it
    has one -- the container's transmission jumps too.

    Parameters
    ----------
    experiment : Experiment
        The configuration.
    low, high : float
        The energy range, in keV.

    Returns
    -------
    list[float]
        Ascending, strictly inside the range.
    """
    edges: set[float] = set()
    built, _ = crystal_material(experiment.crystal.composition)
    if built is not None:
        for element in built.elements:
            edges.update(element.edges)
    container = experiment.crystal.container
    if container is not None:
        if container.type == "mixture":
            edges.update(mixture_edges(container.material))
        else:
            edges.update(elemental_edges(container.elements))
    return sorted(edge for edge in edges if low < edge < high)


def edge_free_intervals(
    low: float,
    high: float,
    edges: Iterable[float],
    clearance: float = _EDGE_CLEARANCE,
) -> list[tuple[float, float]]:
    """Split a range at edges into pieces an objective is smooth on.

    Parameters
    ----------
    low, high : float
        The range.
    edges : Iterable[float]
        Where to split; any outside the range are ignored.
    clearance : float
        How far each piece stops short of an edge that bounds it. A piece
        narrower than that is dropped.

    Returns
    -------
    list[tuple[float, float]]
        Ascending pieces.
    """
    cuts = sorted({edge for edge in edges if low < edge < high})
    bounds = [low, *cuts, high]
    pieces = [
        (
            start if start == low else start + clearance,
            end if end == high else end - clearance,
        )
        for start, end in itertools.pairwise(bounds)
    ]
    return [(start, end) for start, end in pieces if start < end]


def maximise_energy(
    experiment: Experiment,
    objective: Callable[[list[WedgeSummary], Any], Any],
    low: float,
    high: float,
    *,
    edges: Iterable[float] = (),
    tolerance: float = 0.01,
    clearance: float = _EDGE_CLEARANCE,
) -> list[Search]:
    """Search for the beam energy that maximises an objective.

    The range is split at the crystal's and the container's absorption
    edges, and at any ``edges`` the objective itself adds -- a detector's,
    say -- and each piece is searched with :func:`maximise_on_interval`.

    Parameters
    ----------
    experiment : Experiment
        The configuration; everything but the energy is held fixed.
    objective : Callable[[list[WedgeSummary], Any], Any]
        Takes the summaries :func:`solve` returns and the energy, both
        carrying the gradient, and returns the scalar to maximise.
    low, high : float
        The energy range, in keV -- what the beamline can reach.
    edges : Iterable[float], optional
        Further energies at which the objective jumps.
    tolerance : float
        How closely to locate each piece's maximum, in keV.
    clearance : float
        How far to keep from each edge, in keV. The default only keeps the
        search off the edge itself. Absorption edges in the model are sharp
        steps; real ones have fine structure for tens of eV around them, and
        a beam has an energy spread, so a strategy meant for a beamline
        should keep further away.

    Returns
    -------
    list[Search]
        One per piece, in ascending energy. The best energy overall is
        ``max(searches, key=lambda s: s.best[1]).best[0]``.
    """
    walls = [*absorption_edges(experiment, low, high), *edges]

    def at(energy: Any) -> Any:
        return objective(solve(experiment, {"beam.energy": energy}), energy)

    return [
        maximise_on_interval(at, start, end, tolerance=tolerance)
        for start, end in edge_free_intervals(low, high, walls, clearance)
    ]
