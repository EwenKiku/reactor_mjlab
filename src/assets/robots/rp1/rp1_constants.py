"""RoboParty RP1 (24 DoF) entity configuration for mjlab.

Ported from ``robolab/assets/robots/roboparty.py`` (IsaacLab 3, Newton/MJWarp).
The measured RoboParty motor envelope (delayed PD + DC torque-speed curve) maps
onto mjlab's ``DcMotorActuatorCfg``:

* ``effort_limit``      = IsaacLab ``actuator_effort_limit`` (plateau torque).
* ``velocity_limit``    = IsaacLab ``actuator_velocity_limit``.
* ``saturation_effort`` = effort / (1 - keep_velocity / velocity_limit), which is
  exactly how ``RobopartyActuator`` derives its stall torque.
* ``delay_*``           = one constant lag in [0, 2] physics steps per episode,
  shared by all joints (the four groups fuse into one delay buffer).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

RP1_DIR = Path(__file__).resolve().parent
RP1_24DOF_XML = RP1_DIR / "mjcf" / "robot_24dof.xml"

RP1_FOOT_SCAN_SITES: tuple[str, str] = ("left_foot_scan", "right_foot_scan")
RP1_FOOT_GEOM_EXPR = r"^(left|right)_foot_.*_collision$"

##
# MJCF.
##


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(RP1_24DOF_XML))
  # mjlab creates its own actuators and sensors.
  for act in list(spec.actuators):
    spec.delete(act)
  for sensor in list(spec.sensors):
    spec.delete(sensor)
  # Origins for the rough-terrain foot height scanners (IsaacLab offset x=0.025).
  for side, site_name in zip(("left", "right"), RP1_FOOT_SCAN_SITES, strict=True):
    body = spec.body(f"{side}_ankle_roll_link")
    body.add_site(name=site_name, pos=(0.025, 0.0, 0.0), size=(0.005, 0.005, 0.005))
  return spec


##
# Actuators.
##

_GROUPS: dict[str, dict[str, float]] = {
  # IsaacLab regex: ".*(waist|hip_(pitch|roll)|knee).*"
  "big": dict(
    effort=140.0,
    velocity=13.61357,
    keep_velocity=8.37758,
    rated_effort=44.0,
    rated_velocity=8.37758,
    stiffness=120.0,
    damping=7.5,
    armature=0.02908560,
  ),
  # ".*(hip_yaw|shoulder_pitch).*"
  "mid": dict(
    effort=55.0,
    velocity=16.23156,
    keep_velocity=6.28319,
    rated_effort=14.0,
    rated_velocity=6.28319,
    stiffness=35.0,
    damping=2.5,
    armature=0.00884000,
  ),
  # ".*_ankle_(pitch|roll).*"
  "ankle": dict(
    effort=64.0,
    velocity=15.70796,
    keep_velocity=14.66077,
    rated_effort=20.0,
    rated_velocity=6.28319,
    stiffness=25.0,
    damping=1.5,
    armature=0.006266268,
  ),
  # ".*(shoulder_(roll|yaw)|elbow|wrist).*"
  "small": dict(
    effort=32.0,
    velocity=15.70796,
    keep_velocity=14.66077,
    rated_effort=10.0,
    rated_velocity=6.28319,
    stiffness=12.5,
    damping=0.75,
    armature=0.00313313,
  ),
}
_GROUP_JOINTS: dict[str, tuple[str, ...]] = {
  "big": (
    ".*_hip_pitch_joint",
    ".*_hip_roll_joint",
    ".*_knee_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
  ),
  "mid": (".*_hip_yaw_joint", ".*_shoulder_pitch_joint"),
  "ankle": (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  "small": (
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_joint",
  ),
}

# delay_update_period exceeds the 4000 physics steps of a 20 s episode (and the
# 8000 of a 40 s play episode), so the lag sampled after each reset is held for
# the whole episode, matching IsaacLab's DelayedPDActuator.
_DELAY = dict(
  delay_min_lag=0,
  delay_max_lag=2,
  delay_update_period=100_000,
  delay_per_env_phase=False,
  delay_hold_prob=0.0,
)


def _make_actuator(group: str) -> DcMotorActuatorCfg:
  p = _GROUPS[group]
  return DcMotorActuatorCfg(
    target_names_expr=_GROUP_JOINTS[group],
    stiffness=p["stiffness"],
    damping=p["damping"],
    effort_limit=p["effort"],
    saturation_effort=p["effort"] / (1.0 - p["keep_velocity"] / p["velocity"]),
    velocity_limit=p["velocity"],
    armature=p["armature"],
    frictionloss=0.1,
    viscous_damping=0.01,
    **_DELAY,
  )


RP1_ACTUATORS: tuple[DcMotorActuatorCfg, ...] = tuple(
  _make_actuator(g) for g in ("big", "mid", "ankle", "small")
)


##
# Collisions.
##

RP1_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={RP1_FOOT_GEOM_EXPR: 3, ".*": 1},
  priority={RP1_FOOT_GEOM_EXPR: 1, ".*": 0},
)

##
# Initial state.
##

# mjlab resolves patterns first-match-wins, so the catch-all must come last.
RP1_DEFAULT_JOINT_POS: dict[str, float] = {
  "left_hip_pitch_joint": -0.1,
  "left_knee_joint": -0.3,
  "left_ankle_pitch_joint": -0.2,
  "left_shoulder_pitch_joint": 0.2,
  "left_shoulder_roll_joint": 0.2,
  "left_elbow_joint": -1.2,
  "right_hip_pitch_joint": 0.1,
  "right_knee_joint": 0.3,
  "right_ankle_pitch_joint": -0.2,
  "right_shoulder_pitch_joint": -0.2,
  "right_shoulder_roll_joint": -0.2,
  "right_elbow_joint": 1.2,
  ".*": 0.0,
}

RP1_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.78),
  joint_pos=dict(RP1_DEFAULT_JOINT_POS),
  joint_vel={".*": 0.0},
)

RP1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=RP1_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_rp1_robot_cfg() -> EntityCfg:
  """Return a fresh RP1 entity configuration."""
  return EntityCfg(
    init_state=RP1_INIT_STATE,
    collisions=(RP1_COLLISION,),
    spec_fn=get_spec,
    articulation=RP1_ARTICULATION,
    sort_actuators=True,
  )
