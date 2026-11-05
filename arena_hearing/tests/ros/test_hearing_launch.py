"""hearing.launch.py: which hearing nodes an env gets for its robot.hearing.* configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAUNCH = Path(__file__).parents[2] / "launch" / "hearing.launch.py"


@pytest.fixture(autouse=True)
def _launch_gate():
    pytest.importorskip("launch_ros")
    pytest.importorskip("arena_simulation_setup")


def _executables(**configs: str) -> list[str]:
    import launch
    from launch_ros.actions import Node

    spec = importlib.util.spec_from_file_location("hearing_launch", LAUNCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = launch.LaunchContext()
    context.launch_configurations.update({"env.ns": "/arena/env_0", "tg_node": "arena_env_node", "frontend": "bus", "policy": "belief", **configs})
    return [action.node_executable for action in module._hearing(context) if isinstance(action, Node)]


def test_the_search_driver_starts_only_when_enabled() -> None:
    assert "hearing_search_driver" not in _executables()
    assert "hearing_search_driver" in _executables(**{"robot.hearing.driver.enabled": "true"})


def test_a_disabled_search_driver_leaves_the_hearing_nodes_as_they_were() -> None:
    assert _executables(**{"robot.hearing.driver.enabled": "false"}) == ["hearing_belief_node", "hearing_policy"]
