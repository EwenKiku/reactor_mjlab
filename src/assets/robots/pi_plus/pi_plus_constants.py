from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
  
PI_PLUS_DIR = Path(__file__).resolve().parent
PI_PLUS_XML = PI_PLUS_DIR / "xmls" / "pi_plus.xml"

PI_PLUS_FOOT_SCAN_SITES: tuple[str, str] = ("l_foot_scan", "r_foot_scan")
PI_PLUS_FOOT_GEOM_EXPR = r"^(l|r)_(toe|heel)_.*$"



def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PI_PLUS_XML))
  # mjlab creates its own actuators and sensors.
  for act in list(spec.actuators):
    spec.delete(act)
  for sensor in list(spec.sensors):
    spec.delete(sensor)
  for side, site_name in zip(("l", "r"), PI_PLUS_FOOT_SCAN_SITES, strict=True):
    body = spec.body(f"{side}_ankle_roll_link")
    body.add_site(name=site_name, pos=(0.025, 0.0, 0.0), size=(0.005, 0.005, 0.005))
  return spec



def pi_plus_actuator_cfgs() -> tuple[DcMotorActuatorCfg, ...]:
  """Return a fresh tuple of PI PLUS actuator configurations."""
  return (
    DcMotorActuatorCfg(
      target_names_expr=(".*_thigh_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_calf_joint",),
      stiffness=80.0,
      damping=1.1,
      effort_limit=30.0,
      saturation_effort=30.0,
      velocity_limit=17.0,
      armature=0.044277,
      delay_min_lag=1,
      delay_max_lag=2,
      delay_update_period=100_000,
      delay_per_env_phase=False,
      delay_hold_prob=0.0,
    ),
    DcMotorActuatorCfg(
      target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint",),
      stiffness=60.0,
      damping=1.2,
      effort_limit=20.0,
      saturation_effort=20.0,
      velocity_limit=17.0,
      armature=0.044277,
      delay_min_lag=1,
      delay_max_lag=2,
      delay_update_period=100_000,
      delay_per_env_phase=False,
      delay_hold_prob=0.0,
    ),
    DcMotorActuatorCfg(
      target_names_expr=(".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_upper_arm_joint", ".*_elbow_joint",),
      stiffness=30.0,
      damping=0.6,
      effort_limit=20.0,
      saturation_effort=20.0,
      velocity_limit=17.0,
      armature=0.008419,   
      delay_min_lag=1,
      delay_max_lag=2,
      delay_update_period=100_000,
      delay_per_env_phase=False,
      delay_hold_prob=0.0,
    ),
    DcMotorActuatorCfg(
      target_names_expr=("head_yaw_joint", "head_pitch_joint",),
      stiffness=10.0,
      damping=0.6,
      effort_limit=20.0,
      saturation_effort=20.0,
      velocity_limit=17.0,
      armature=0.008419,   
      delay_min_lag=1,
      delay_max_lag=2,
      delay_update_period=100_000,
      delay_per_env_phase=False,
      delay_hold_prob=0.0,
    ),
  )


PI_PLUS_DEFAULT_JOINT_POS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.25,
    ".*_calf_joint": 0.65,
    ".*_ankle_pitch_joint": -0.4,
    ".*_elbow_joint": -1.2,
    "l_shoulder_roll_joint": 0.2,
    "l_shoulder_pitch_joint": 0.0,
    "r_shoulder_roll_joint": -0.2,
    "r_shoulder_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
}

PI_PLUS_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.351),
  joint_pos=PI_PLUS_DEFAULT_JOINT_POS,
  joint_vel={".*": 0.0},
)

PI_PLUS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=pi_plus_actuator_cfgs(),
  soft_joint_pos_limit_factor=0.9,
)


def get_pi_plus_robot_cfg() -> EntityCfg:
  """Return a fresh PI PLUS entity configuration."""
  return EntityCfg(
    init_state=PI_PLUS_INIT_STATE,
    spec_fn=get_spec,
    articulation=PI_PLUS_ARTICULATION,
  )
