from __future__ import annotations

import math

import numpy as np

from arena_hearing.search.frontiers import FREE, UNKNOWN, Frontier, MapGrid, Opening
from arena_hearing.search.goals import BeliefGoal, BeliefMap, GoalView, Heard, NearestGoal

RESOLUTION = 0.1
ROBOT = (2.0, 2.0)
NEAR = (1.0, 4.0)
FAR = (6.0, 4.0)


def _grid() -> MapGrid:
    data = np.full((60, 80), UNKNOWN, dtype=np.int16)
    data[:40, :] = FREE
    return MapGrid(data=data, origin=(0.0, 0.0), resolution=RESOLUTION, occupied_threshold=50)


def _frontier(xy: tuple[float, float]) -> Frontier:
    return Frontier(cells_xy=np.array([xy]), goal_xy=xy)


def _belief(*peaks: tuple[float, float]) -> BeliefMap:
    data = np.zeros((60, 80), dtype=np.float32)
    for x, y in peaks:
        data[int(y / RESOLUTION), int(x / RESOLUTION)] = 1.0
    return BeliefMap(data=data, origin=(0.0, 0.0), resolution=RESOLUTION)


def _view(belief: BeliefMap | None = None, heard: Heard | None = None, openings: list[Opening] | None = None) -> GoalView:
    return GoalView(robot_xy=ROBOT, grid=_grid(), frontiers=[_frontier(NEAR), _frontier(FAR)], openings=openings or [], belief=belief, heard=heard)


def _rule(snap: bool = False) -> BeliefGoal:
    return BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=2.0, distance_scale_m=5.0, snap=snap, snap_max_error_rad=math.radians(20.0))


def test_nearest_rule_takes_the_closest_frontier() -> None:
    assert NearestGoal().choose(_view()) == NEAR


def test_belief_rule_prefers_the_frontier_facing_the_call() -> None:
    assert _rule().choose(_view(belief=_belief((6.5, 4.5)))) == FAR


def test_belief_rule_without_belief_mass_takes_the_nearest_frontier() -> None:
    assert _rule().choose(_view(belief=_belief())) == NEAR


def test_belief_rule_goes_to_a_located_person_in_mapped_space() -> None:
    rule = BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=0.9, distance_scale_m=5.0, snap=False, snap_max_error_rad=0.0)
    x, y = rule.choose(_view(belief=_belief((3.05, 1.05))))
    assert math.dist((x, y), (3.05, 1.05)) < 0.1


def test_belief_rule_clamps_a_person_in_unknown_space_to_the_nearest_frontier() -> None:
    rule = BeliefGoal(radius_m=1.5, min_mass=0.5, locate_threshold=0.9, distance_scale_m=5.0, snap=False, snap_max_error_rad=0.0)
    assert rule.choose(_view(belief=_belief((6.05, 5.05)))) == FAR


def test_snap_sends_the_belief_rule_through_the_opening_the_call_came_through() -> None:
    door = Opening(xy=(2.0, 3.5))
    heard = Heard(robot_xy=ROBOT, bearing_rad=math.radians(95.0))
    assert _rule(snap=True).choose(_view(belief=_belief((6.5, 4.5)), heard=heard, openings=[door])) == door.xy


def test_without_snap_the_heard_bearing_is_ignored() -> None:
    heard = Heard(robot_xy=ROBOT, bearing_rad=math.radians(95.0))
    assert _rule().choose(_view(belief=_belief((6.5, 4.5)), heard=heard, openings=[Opening(xy=(2.0, 3.5))])) == FAR
