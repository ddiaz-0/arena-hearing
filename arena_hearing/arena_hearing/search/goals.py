"""Goal rules: which frontier, opening or located person the search drives to next."""

from __future__ import annotations

import math
from typing import Protocol

import attrs
import numpy as np

from arena_hearing.search.frontiers import Frontier, MapGrid, Opening
from arena_hearing.search.snap import snap_bearing


@attrs.frozen
class BeliefMap:
    """A normalised belief grid in [0, 1], on its own layout."""

    data: np.ndarray
    origin: tuple[float, float]
    resolution: float

    def mass_near(self, xy: tuple[float, float], radius_m: float) -> float:
        h, w = self.data.shape
        xs = self.origin[0] + (np.arange(w) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(h) + 0.5) * self.resolution
        inside = (xs[None, :] - xy[0]) ** 2 + (ys[:, None] - xy[1]) ** 2 <= radius_m * radius_m
        return float(self.data[inside].sum())

    def peak(self) -> tuple[tuple[float, float], float]:
        """Center and value of the maximum cell."""
        row, col = divmod(int(np.argmax(self.data)), self.data.shape[1])
        return (self.origin[0] + (col + 0.5) * self.resolution, self.origin[1] + (row + 0.5) * self.resolution), float(self.data[row, col])


@attrs.frozen
class Heard:
    """The latest call: the robot position it was heard from and its map-frame bearing."""

    robot_xy: tuple[float, float]
    bearing_rad: float


@attrs.frozen
class GoalView:
    robot_xy: tuple[float, float]
    grid: MapGrid
    frontiers: list[Frontier]
    openings: list[Opening]
    belief: BeliefMap | None
    heard: Heard | None


class GoalRule(Protocol):
    def choose(self, view: GoalView) -> tuple[float, float] | None:
        """The next goal, None when nothing is left to explore."""
        ...


class NearestGoal:
    """The closest frontier, the no-audio baseline."""

    def choose(self, view: GoalView) -> tuple[float, float] | None:
        if not view.frontiers:
            return None
        return min((f.goal_xy for f in view.frontiers), key=lambda xy: math.dist(xy, view.robot_xy))


@attrs.frozen(kw_only=True)
class BeliefGoal:
    """A located person, else the opening a call snapped to, else the frontier with the most belief mass nearby per distance."""

    radius_m: float
    min_mass: float
    locate_threshold: float
    distance_scale_m: float
    snap: bool
    snap_max_error_rad: float

    def choose(self, view: GoalView) -> tuple[float, float] | None:
        if view.belief is not None:
            peak, value = view.belief.peak()
            if value >= self.locate_threshold:
                return _clamp(view, peak)
        if self.snap and view.heard is not None:
            candidates = [o.xy for o in view.openings] + [f.goal_xy for f in view.frontiers]
            index = snap_bearing(view.heard.robot_xy, view.heard.bearing_rad, candidates, self.snap_max_error_rad)
            if index is not None:
                return candidates[index]
        if view.belief is not None and view.frontiers:
            belief = view.belief
            masses = [belief.mass_near(f.goal_xy, self.radius_m) for f in view.frontiers]
            if max(masses) >= self.min_mass:
                scores = [m / (1.0 + math.dist(f.goal_xy, view.robot_xy) / self.distance_scale_m) for m, f in zip(masses, view.frontiers, strict=True)]
                return view.frontiers[int(np.argmax(scores))].goal_xy
        return NearestGoal().choose(view)


def _clamp(view: GoalView, xy: tuple[float, float]) -> tuple[float, float] | None:
    """``xy`` when it is mapped free space, else the nearest frontier, since Nav2 rejects goals off the map."""
    grid = view.grid
    col = int(math.floor((xy[0] - grid.origin[0]) / grid.resolution))
    row = int(math.floor((xy[1] - grid.origin[1]) / grid.resolution))
    h, w = grid.data.shape
    if 0 <= row < h and 0 <= col < w and grid.free[row, col]:
        return xy
    if not view.frontiers:
        return None
    return min((f.goal_xy for f in view.frontiers), key=lambda goal: math.dist(goal, xy))
