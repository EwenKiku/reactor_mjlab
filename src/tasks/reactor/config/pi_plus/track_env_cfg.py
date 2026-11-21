"""Pi Plus downstream motion tracking on ReActor-retargeted clips.

This is the second half of the paper: once the retargeting parameters ``p``
converge, the reference motions they produce are physically consistent, and a
plain DeepMimic-style tracking policy is trained on them **without** residual
forces or domain randomisation.  The reward follows Table 6, which differs from
the retargeting reward of Table 1 in its weights (root xy is 5.0 instead of 2.0,
survival 10.0, joint action rate 0.15) and has no wrench terms.

Data comes from ``tools/export_reactor_retarget.py``: ``joint_pos`` arrived from
IK against the retargeted body targets, and the body trajectories are the
retargeted ``g_t`` themselves.
"""

from __future__ import annotations

from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from src.tasks.reactor.mdp.track_commands import TrackingCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.assets.robots.pi_plus.pi_plus_constants import (
  PI_PLUS_FOOT_GEOM_EXPR,
  PI_PLUS_JOINT_ORDER,
  PI_PLUS_RATED_EFFORT,
  PI_PLUS_RATED_VELOCITY,
  get_pi_plus_robot_cfg,
)
from src.tasks.base import mdp as base_mdp
from src.tasks.reactor.mdp.tracking import (
  body_orientation,
  body_position,
  reference_joint_target,
  reference_progress,
  root_ang_vel,
  root_height,
  root_lin_vel,
  root_orientation,
  root_position_xy,
)

# config/pi_plus/<file> -> repo root is five levels up.
_REPO_ROOT = Path(__file__).resolve().parents[5]
MOTION_DIR = _REPO_ROOT / "src" / "assets" / "motions" / "reactor" / "pi_plus"

ANCHOR_BODY = "base_link"
TRACKING_BODY_NAMES = (
  "base_link",
  "torso_link",
  "head_pitch_link",
  "l_shoulder_roll_link",
  "l_upper_arm_link",
  "l_elbow_link",
  "l_wrist_link",
  "r_shoulder_roll_link",
  "r_upper_arm_link",
  "r_elbow_link",
  "r_wrist_link",
  "l_thigh_link",
  "l_calf_link",
  "l_ankle_roll_link",
  "r_thigh_link",
  "r_calf_link",
  "r_ankle_roll_link",
)
EE_BODY_NAMES = (
  "l_ankle_roll_link",
  "r_ankle_roll_link",
  "l_wrist_link",
  "r_wrist_link",
)

FEET = SceneEntityCfg("robot", body_names=("l_ankle_roll_link", "r_ankle_roll_link"), preserve_order=True)
ALL_JOINTS = SceneEntityCfg("robot", joint_names=(".*",))
MOTION = {"command_name": "motion"}

VELOCITY_RANGE = {
  "x": (-0.3, 0.3),
  "y": (-0.3, 0.3),
  "z": (-0.1, 0.1),
  "roll": (-0.26, 0.26),
  "pitch": (-0.26, 0.26),
  "yaw": (-0.39, 0.39),
}
POSE_RANGE = {
  "x": (-0.03, 0.03),
  "y": (-0.03, 0.03),
  "z": (-0.01, 0.01),
  "roll": (-0.05, 0.05),
  "pitch": (-0.05, 0.05),
  "yaw": (-0.1, 0.1),
}


def _observations() -> dict[str, ObservationGroupCfg]:
  def un(x: float) -> Unoise:
    return Unoise(n_min=-x, n_max=x)

  actor = {
    "reference_joint_target": ObservationTermCfg(
      func=reference_joint_target, params=MOTION
    ),
    "reference_progress": ObservationTermCfg(
      func=reference_progress, params=MOTION
    ),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, noise=un(0.2)),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=un(0.05)
    ),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, noise=un(0.01)),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, noise=un(0.5)),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic = {
    "command": ObservationTermCfg(func=envs_mdp.generated_commands, params=MOTION),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=tracking_mdp.motion_anchor_pos_b, params=MOTION
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=tracking_mdp.motion_anchor_ori_b, params=MOTION
    ),
    "robot_body_pos_b": ObservationTermCfg(
      func=tracking_mdp.robot_body_pos_b, params=MOTION
    ),
    "robot_body_ori_b": ObservationTermCfg(
      func=tracking_mdp.robot_body_ori_b, params=MOTION
    ),
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  return {
    "actor": ObservationGroupCfg(terms=actor, enable_corruption=True),
    "critic": ObservationGroupCfg(terms=critic, enable_corruption=False),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  """Table 6: downstream tracking weights, no wrench terms."""

  def track(func, weight: float, std: float) -> RewardTermCfg:
    return RewardTermCfg(func=func, weight=weight, params={**MOTION, "std": std})

  def survival(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

  return {
    "root_position_xy": track(root_position_xy, 5.0, 0.3),
    "root_height": track(root_height, 5.0, 0.2),
    "root_orientation": track(root_orientation, 3.0, 0.4),
    "root_lin_vel": track(root_lin_vel, 0.5, 1.0),
    "root_ang_vel": track(root_ang_vel, 0.5, 3.14),
    "body_position": track(body_position, 5.0, 0.3),
    "body_orientation": track(body_orientation, 2.5, 0.4),
    "survival": RewardTermCfg(func=survival, weight=10.0),
    "joint_torques_l2": RewardTermCfg(func=base_mdp.joint_torques_l2, weight=-1e-4),
    "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-8),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.15),
    "action_jerk_l2": RewardTermCfg(func=base_mdp.action_jerk_l2, weight=-1e-2),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": ALL_JOINTS}
    ),
    "joint_rated_velocity": RewardTermCfg(
      func=base_mdp.joint_rated_velocity_limit,
      weight=-0.05,
      params={"rated_velocity": PI_PLUS_RATED_VELOCITY},
    ),
    "joint_rated_effort": RewardTermCfg(
      func=base_mdp.joint_rated_effort_limit,
      weight=-0.04,
      params={"rated_effort": PI_PLUS_RATED_EFFORT},
    ),
    "feet_slide": RewardTermCfg(
      func=base_mdp.feet_slide,
      weight=-0.3,
      params={"sensor_name": "feet_contact", "asset_cfg": FEET},
    ),
    "undesired_contacts": RewardTermCfg(
      func=base_mdp.undesired_self_contacts,
      weight=-0.5,
      params={"sensor_name": "self_contact"},
    ),
  }


def _events(play: bool) -> dict[str, EventTermCfg]:
  events = {
    "foot_friction": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=(PI_PLUS_FOOT_GEOM_EXPR,)),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
  }
  # The paper's downstream policies are trained without domain randomisation,
  # but a mild push keeps the policy from overfitting to a perfect reference.
  if not play:
    events["push_robot"] = EventTermCfg(
      mode="interval",
      func=envs_mdp.push_by_setting_velocity,
      interval_range_s=(4.0, 8.0),
      params={"velocity_range": dict(VELOCITY_RANGE)},
    )
  return events


def _scene(envs: int) -> SceneCfg:
  return SceneCfg(
    num_envs=envs,
    env_spacing=2.5,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_pi_plus_robot_cfg()},
    sensors=(
      ContactSensorCfg(
        name="feet_contact",
        primary=ContactMatch(
          mode="body",
          pattern=("l_ankle_roll_link", "r_ankle_roll_link"),
          entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=3,
      ),
      ContactSensorCfg(
        name="self_contact",
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern=ANCHOR_BODY, entity="robot"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=3,
      ),
    ),
  )


def pi_plus_track_env_cfg(
  play: bool = False, motion_file: str | None = None, num_envs: int = 4096
) -> ManagerBasedRlEnvCfg:
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    episode_length_s=10.0,
    scene=_scene(1 if play else num_envs),
    observations=_observations(),
    actions={
      "joint_pos": envs_mdp.actions.JointPositionActionCfg(
        entity_name="robot",
        actuator_names=PI_PLUS_JOINT_ORDER,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
      )
    },
    commands={
      "motion": TrackingCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=not play,
        motion_files=(motion_file or str(MOTION_DIR),),
        anchor_body_name=ANCHOR_BODY,
        body_names=TRACKING_BODY_NAMES,
        pose_range=dict(POSE_RANGE),
        velocity_range=dict(VELOCITY_RANGE),
        joint_position_range=(-0.1, 0.1),
        adaptive_kernel_size=2,
      )
    },
    events=_events(play),
    rewards=_rewards(),
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "anchor_pos": TerminationTermCfg(
        func=tracking_mdp.bad_anchor_pos_z_only,
        params={**MOTION, "threshold": 0.25},
      ),
      "anchor_ori": TerminationTermCfg(
        func=tracking_mdp.bad_anchor_ori,
        params={**MOTION, "threshold": 0.8, "asset_cfg": SceneEntityCfg("robot")},
      ),
      "ee_body_pos": TerminationTermCfg(
        func=tracking_mdp.bad_motion_body_pos_z_only,
        params={**MOTION, "threshold": 0.25, "body_names": EE_BODY_NAMES},
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=ANCHOR_BODY,
      distance=2.5,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=500,
      nconmax=256,
      mujoco=MujocoCfg(timestep=0.005, iterations=50, ls_iterations=100),
    ),
  )
  if play:
    cfg.episode_length_s = int(1e6)
    cfg.observations["actor"].enable_corruption = False
    motion = cfg.commands["motion"]
    assert isinstance(motion, TrackingCommandCfg)
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    motion.sampling_mode = "start"
  return cfg