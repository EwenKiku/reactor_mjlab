"""RP1 ReActor environment. Link map and frame rotations come from calibration.yaml."""

from __future__ import annotations

from pathlib import Path

from src.assets.robots.rp1.rp1_constants import RP1_24DOF_XML, get_rp1_robot_cfg
from src.tasks.reactor.calibration import load_calibration
from src.tasks.reactor.config.reactor_env import reactor_env_cfg

CALIBRATION = load_calibration(Path(__file__).with_name("calibration.yaml"))


def rp1_reactor_env_cfg(
  play: bool = False, motion_file: str | None = None, num_envs: int = 4096
):
  return reactor_env_cfg(
    robot_cfg=get_rp1_robot_cfg(),
    calibration=CALIBRATION,
    xml_path=str(RP1_24DOF_XML),
    limb_bodies=("left_.*_link", "right_.*_link"),
    play=play,
    motion_file=motion_file,
    num_envs=num_envs,
  )
