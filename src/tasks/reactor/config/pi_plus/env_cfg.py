"""Pi Plus ReActor environment. Link map and frame rotations come from calibration.yaml."""

from __future__ import annotations

from pathlib import Path

from src.assets.robots.pi_plus.pi_plus_constants import (
  PI_PLUS_XML,
  get_pi_plus_robot_cfg,
)
from src.tasks.reactor.calibration import load_calibration
from src.tasks.reactor.config.reactor_env import MOTION_DIR, reactor_env_cfg

CALIBRATION = load_calibration(Path(__file__).with_name("calibration.yaml"))


def pi_plus_reactor_env_cfg(
  play: bool = False, motion_file: str | None = None, num_envs: int = 4096
):
  return reactor_env_cfg(
    robot_cfg=get_pi_plus_robot_cfg(),
    calibration=CALIBRATION,
    xml_path=str(PI_PLUS_XML),
    limb_bodies=("l_.*_link", "r_.*_link"),
    play=play,
    motion_file=motion_file,
    num_envs=num_envs,
  )
