from __future__ import annotations

import numpy as np

from arena_hearing.search.driver import Cancel, DriverConfig, NavStatus, Phase, SearchDriver, Send, Tick
from arena_hearing.search.frontiers import FREE, UNKNOWN, MapGrid
from arena_hearing.search.goals import GoalRule, NearestGoal
from arena_hearing.search.stops import EveryGoalStop, NeverStop, StopRule

RESOLUTION = 0.1
ROBOT = (2.0, 2.0)
LISTEN_S = 3.0


def _grid(left_unknown: bool = True, right_unknown: bool = True) -> MapGrid:
    data = np.full((40, 60), FREE, dtype=np.int16)
    if left_unknown:
        data[:, :5] = UNKNOWN
    if right_unknown:
        data[:, 55:] = UNKNOWN
    return MapGrid(data=data, origin=(0.0, 0.0), resolution=RESOLUTION, occupied_threshold=50)


def _driver(stop: StopRule | None = None, goal: GoalRule | None = None) -> SearchDriver:
    config = DriverConfig(listen_s=LISTEN_S, min_frontier_cells=5, reached_radius_m=0.5, opening_min_width_m=0.6, opening_max_width_m=1.6, opening_depth_m=0.6)
    return SearchDriver(goal or NearestGoal(), stop or NeverStop(), config)


def _tick(now: float, status: NavStatus = NavStatus.ACTIVE, grid: MapGrid | None = None, robot_xy: tuple[float, float] = ROBOT) -> Tick:
    return Tick(now=now, robot_xy=robot_xy, grid=grid or _grid(), belief=None, heard=None, calls=0, plan_xy=None, status=status)


def test_the_driver_sends_the_goal_its_rule_chooses() -> None:
    action = _driver().step(_tick(0.0, NavStatus.IDLE))
    assert isinstance(action, Send)
    assert action.xy[0] < 1.0


def test_a_stop_cancels_the_goal_and_listens_for_listen_s() -> None:
    driver = _driver(stop=EveryGoalStop())
    assert isinstance(driver.step(_tick(0.0, NavStatus.IDLE)), Send)
    assert isinstance(driver.step(_tick(0.1)), Cancel)
    assert driver.phase is Phase.LISTEN
    assert driver.step(_tick(0.1 + LISTEN_S - 0.05, NavStatus.IDLE)) is None
    assert isinstance(driver.step(_tick(0.1 + LISTEN_S, NavStatus.IDLE)), Send)
    assert driver.step(_tick(0.2 + LISTEN_S)) is None
    assert driver.phase is Phase.DRIVE


def test_a_failed_goal_is_not_chosen_again() -> None:
    driver = _driver()
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    second = driver.step(_tick(1.0, NavStatus.FAILED))
    assert isinstance(first, Send)
    assert isinstance(second, Send)
    assert second.xy[0] > 5.0


def test_a_reached_goal_selects_the_next_frontier() -> None:
    driver = _driver()
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    assert isinstance(first, Send)
    second = driver.step(_tick(1.0, NavStatus.SUCCEEDED, robot_xy=first.xy))
    assert isinstance(second, Send)
    assert second.xy[0] > 5.0


def test_a_goal_lost_while_driving_is_sent_again() -> None:
    driver = _driver()
    first = driver.step(_tick(0.0, NavStatus.IDLE))
    again = driver.step(_tick(1.0, NavStatus.IDLE))
    assert isinstance(first, Send)
    assert isinstance(again, Send)
    assert again.xy == first.xy


def test_the_driver_is_done_when_no_frontier_is_left() -> None:
    driver = _driver()
    assert driver.step(_tick(0.0, NavStatus.IDLE, grid=_grid(left_unknown=False, right_unknown=False))) is None
    assert driver.phase is Phase.DONE


def test_the_state_counts_stops_and_listening_time() -> None:
    driver = _driver(stop=EveryGoalStop())
    driver.step(_tick(0.0, NavStatus.IDLE))
    driver.step(_tick(0.1))
    driver.step(_tick(0.1 + LISTEN_S, NavStatus.IDLE))
    state = driver.state(0.1 + LISTEN_S)
    assert state["stops"] == 1
    assert abs(state["listened_s"] - LISTEN_S) < 1e-6


def test_a_reset_forgets_failed_goals_and_stops() -> None:
    driver = _driver(stop=EveryGoalStop())
    driver.step(_tick(0.0, NavStatus.IDLE))
    driver.step(_tick(0.1))
    driver.reset()
    assert driver.state(0.2)["stops"] == 0
    action = driver.step(_tick(0.2, NavStatus.IDLE))
    assert isinstance(action, Send)
    assert action.xy[0] < 1.0
