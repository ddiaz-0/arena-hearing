"""hearing_belief and hearing_policy as real processes, bound to a published fleet in a private namespace."""

from __future__ import annotations

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

TG = "task_generator_node"
MODEL = "jackal"
STARTUP_S = 60.0


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("scipy")
    pytest.importorskip("geometry_msgs.msg")
    pytest.importorskip("nav_msgs.msg")
    pytest.importorskip("std_msgs.msg")
    pytest.importorskip("tf2_msgs.msg")
    pytest.importorskip("arena_robots_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")
    pytest.importorskip("visualization_msgs.msg")


def _param_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return "[" + ",".join(_param_text(item) for item in value) + "]"
    return str(value)


@contextlib.contextmanager
def _hearing_process(module: str, ns: str, params: dict[str, object], log: Path) -> Iterator[subprocess.Popen]:
    args = [sys.executable, "-c", f"from arena_hearing.{module} import main; main()", "--ros-args", "-r", f"__ns:={ns}", "-r", f"/tf:={ns}/tf", "-r", f"/tf_static:={ns}/tf_static"]
    for name, value in params.items():
        args += ["-p", f"{name}:={_param_text(value)}"]
    with log.open("w") as out:
        proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            yield proc
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)


class _Rig:
    """Test-side node in the namespace of one hearing process."""

    def __init__(self, ns: str, proc: subprocess.Popen, log: Path) -> None:
        import rclpy.node

        self.ns = ns
        self.tg = f"{ns}/{TG}"
        self.proc = proc
        self.log = log
        self.node = rclpy.node.Node(f"probe_{uuid.uuid4().hex[:8]}", namespace=ns)

    def spin_until(self, predicate: Callable[[], bool], timeout_s: float, what: str, tick: Callable[[], None] | None = None) -> None:
        import rclpy

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if tick is not None:
                tick()
            if predicate():
                return
            assert self.proc.poll() is None, f"node exited with {self.proc.returncode}:\n{self.log.read_text()[-4000:]}"
        raise AssertionError(f"timed out waiting for {what}:\n{self.log.read_text()[-4000:]}")

    def destroy(self) -> None:
        self.node.destroy_node()


@contextlib.contextmanager
def _rig(module: str, params: dict[str, object], tmp_path: Path) -> Iterator[_Rig]:
    ns = f"/t_{uuid.uuid4().hex[:8]}"
    log = tmp_path / f"{module}.log"
    with _hearing_process(module, ns, params, log) as proc:
        rig = _Rig(ns, proc, log)
        try:
            yield rig
        finally:
            rig.destroy()


def _prefix(rig: _Rig, name: str) -> str:
    return f"{rig.ns.strip('/')}/{name}"


def _fleet(rig: _Rig, names: tuple[str, ...]):
    from task_generator_msgs.msg import RobotFleet, RobotState

    fleet = RobotFleet()
    for name in names:
        state = RobotState()
        state.descriptor.name = name
        state.descriptor.model = MODEL
        state.descriptor.ns = f"{rig.ns}/{name}"
        state.descriptor.frame = _prefix(rig, name)
        fleet.robots.append(state)
    return fleet


def _base_frame(rig: _Rig, name: str) -> str:
    from arena_robots.fleet import RobotBinding

    return RobotBinding.resolve(name=name, model=MODEL, namespace=f"{rig.ns}/{name}", frame_prefix=_prefix(rig, name)).base_frame


def _publish_fleet(rig: _Rig, names: tuple[str, ...]) -> None:
    from arena_rclpy_mixins import qos
    from task_generator_msgs.msg import RobotFleet

    pub = rig.node.create_publisher(RobotFleet, f"{rig.tg}/state/robots", qos.latched())
    pub.publish(_fleet(rig, names))


def _publish_tf(rig: _Rig, child: str, x: float, y: float, yaw: float = 0.0) -> None:
    from arena_rclpy_mixins import qos
    from geometry_msgs.msg import TransformStamped
    from tf2_msgs.msg import TFMessage

    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = child
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.z = math.sin(yaw / 2.0)
    transform.transform.rotation.w = math.cos(yaw / 2.0)
    pub = rig.node.create_publisher(TFMessage, f"{rig.ns}/tf_static", qos.latched())
    pub.publish(TFMessage(transforms=[transform]))


def _detection(frame_id: str, azimuth_rad: float, level_db: float = 50.0, kind: str = "footstep"):
    from arena_robots_msgs.msg import SoundDetection

    msg = SoundDetection()
    msg.header.frame_id = frame_id
    msg.frontend = "bus"
    msg.kind = kind
    msg.azimuth_rad = azimuth_rad
    msg.elevation_rad = math.nan
    msg.level_db = level_db
    msg.confidence = 1.0
    return msg


def _detection_publisher(rig: _Rig, name: str):
    from arena_rclpy_mixins import qos
    from arena_robots_msgs.msg import SoundDetection

    pub = rig.node.create_publisher(SoundDetection, f"{rig.tg}/{name}/hearing/bus/detections", qos.reliable(50))
    rig.spin_until(lambda: pub.get_subscription_count() > 0, STARTUP_S, f"hearing_belief to bind {name}")
    return pub


def _grids(rig: _Rig, name: str) -> list:
    from arena_rclpy_mixins import qos
    from nav_msgs.msg import OccupancyGrid

    received: list[OccupancyGrid] = []
    rig.node.create_subscription(OccupancyGrid, f"{rig.tg}/{name}/hearing/belief_grid", received.append, qos.reliable(10))
    return received


def _as_array(grid) -> np.ndarray:
    return np.asarray(grid.data, dtype=np.int8).reshape(grid.info.height, grid.info.width)


def _belief_params(**given: object) -> dict[str, object]:
    return {
        "belief.standalone.enabled": True,
        "belief.standalone.origin_m": [-10.0, -10.0],
        "belief.standalone.size_m": [20.0, 20.0],
        "belief.markers.enabled": False,
        **given,
    }


@pytest.mark.usefixtures("default_sounds")
def test_belief_node_paints_the_wedge_of_each_robot_along_its_bearing_and_not_behind(tmp_path: Path) -> None:
    with _rig("belief_node", _belief_params(), tmp_path) as rig:
        _publish_fleet(rig, ("r0", "r1"))
        _publish_tf(rig, _base_frame(rig, "r0"), 0.0, 0.0)
        _publish_tf(rig, _base_frame(rig, "r1"), 3.0, 3.0)
        r0_grids, r1_grids = _grids(rig, "r0"), _grids(rig, "r1")
        pub = _detection_publisher(rig, "r0")
        _detection_publisher(rig, "r1")
        rig.spin_until(lambda: bool(r0_grids) and int(_as_array(r0_grids[-1]).sum()) > 0, 10.0, "a painted r0 belief grid", tick=lambda: pub.publish(_detection("map", 0.0)))
        rig.spin_until(lambda: len(r1_grids) >= 3, 10.0, "r1 belief grids")

        grid_msg = r0_grids[-1]
        info = grid_msg.info
        grid = _as_array(grid_msg)
        origin_col = int(round((0.0 - info.origin.position.x) / info.resolution))
        assert grid[:, :origin_col].sum() == 0
        assert grid[:, origin_col:].sum() > 0
        assert all(int(_as_array(msg).sum()) == 0 for msg in r1_grids)


def _grid_msg(size: int, resolution: float, origin: float, data: np.ndarray):
    from nav_msgs.msg import OccupancyGrid

    msg = OccupancyGrid()
    msg.header.frame_id = "map"
    msg.info.resolution = resolution
    msg.info.width = size
    msg.info.height = size
    msg.info.origin.position.x = origin
    msg.info.origin.position.y = origin
    msg.info.origin.orientation.w = 1.0
    msg.data = np.asarray(data, dtype=np.int8).reshape(-1).tolist()
    return msg


class _PolicyIo:
    """Map, belief and plan in, mask and state out, for robot r0."""

    def __init__(self, rig: _Rig) -> None:
        from arena_rclpy_mixins import qos
        from nav_msgs.msg import OccupancyGrid, Path
        from std_msgs.msg import String

        self.rig = rig
        self.masks: list[OccupancyGrid] = []
        self.states: list[dict] = []
        self.map_pub = rig.node.create_publisher(OccupancyGrid, f"{rig.tg}/map", qos.latched())
        self.belief_pub = rig.node.create_publisher(OccupancyGrid, f"{rig.tg}/r0/hearing/belief_grid", qos.reliable(1))
        self.plan_pub = rig.node.create_publisher(Path, f"{rig.tg}/r0/plan", qos.reliable(1))
        rig.node.create_subscription(OccupancyGrid, f"{rig.tg}/r0/hearing/speed_filter_mask", self.masks.append, qos.latched())
        rig.node.create_subscription(String, f"{rig.tg}/r0/hearing/policy_state", lambda msg: self.states.append(json.loads(msg.data)), qos.latched())
        rig.spin_until(lambda: self.belief_pub.get_subscription_count() > 0 and self.plan_pub.get_subscription_count() > 0, STARTUP_S, "hearing_policy to bind r0")


def test_policy_node_cruise_speed_mask_around_blob(tmp_path: Path) -> None:
    params = {"policy.max_linear_mps": 1.0, "policy.listen.enabled": False, "policy.yield.enabled": False}
    with _rig("policy_node", params, tmp_path) as rig:
        _publish_fleet(rig, ("r0",))
        _publish_tf(rig, _base_frame(rig, "r0"), 0.0, 0.0)
        io = _PolicyIo(rig)

        size, resolution, origin = 100, 0.1, -5.0
        blob_xy = (2.0, 0.0)
        centers = origin + (np.arange(size) + 0.5) * resolution
        xx, yy = np.meshgrid(centers, centers)
        belief = _grid_msg(size, resolution, origin, np.where(np.hypot(xx - blob_xy[0], yy - blob_xy[1]) <= 0.3, 100, 0))
        io.map_pub.publish(_grid_msg(size, resolution, origin, np.zeros((size, size))))
        rig.spin_until(lambda: bool(io.masks) and bool(io.states), 10.0, "a mask and a state", tick=lambda: io.belief_pub.publish(belief))

        mask = _as_array(io.masks[-1])

        def cell(x: float, y: float) -> tuple[int, int]:
            return int((y - origin) / resolution), int((x - origin) / resolution)

        assert mask[cell(*blob_xy)] == 40
        assert mask[cell(-4.0, -4.0)] == 0
        assert io.states[-1]["state"] == "cruise"
