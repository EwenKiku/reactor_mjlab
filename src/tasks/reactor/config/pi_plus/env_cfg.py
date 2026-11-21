"""Pi Plus ReActor physics-aware retargeting environment.

The policy tracks the *parameterized* reference ``g_t`` built from a SOMA
(BONES-SEED) clip, while the upper level simultaneously refines the retargeting
parameters ``p`` so the reference and the simulated motion agree.  This is the
bilevel problem of Eq. (1); the policy is the lower level, ``p`` the upper one.

Observation groups
------------------
``actor``    local reference body poses/velocities and proprioception (Sec. 6).
``critic``   includes simulated body positions, clip progress and the residual wrench.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.assets.robots.pi_plus.pi_plus_constants import (
  PI_PLUS_FOOT_GEOM_EXPR,
  PI_PLUS_JOINT_ORDER,
  get_pi_plus_robot_cfg,
)
from src.tasks.reactor import mdp as reactor_mdp
from src.tasks.reactor.correspondences import (
  ROOT_TARGET,
  SOURCE_BODIES,
  TARGET_BODIES,
  nominal_robot_frames,
)
from src.tasks.reactor.mdp.actions import RootWrenchActionCfg
from src.tasks.reactor.mdp.commands import RetargetCommandCfg

# config/pi_plus/<file> -> repo root is five levels up.
_REPO_ROOT = Path(__file__).resolve().parents[5]
MOTION_DIR = _REPO_ROOT / "src" / "assets" / "bones_seed_motions" / "train"
PI_PLUS_XML = _REPO_ROOT / "src" / "assets" / "robots" / "pi_plus" / "xmls" / "pi_plus.xml"

#: Bodies whose reference targets the policy cannot reach honestly; they get the
#: swing-only rotation error (paper Sec. 5).
SWING_ONLY_BODIES = ("torso_link",)


def _target_nominal(env) -> tuple:
  """Nominal Pi Plus frames for the correspondences (Eq. 8-9 input)."""
  return nominal_robot_frames(str(PI_PLUS_XML), TARGET_BODIES)


MOTION = {"command_name": "motion"}

WRENCH_CFG = RootWrenchActionCfg(
  entity_name="robot",
  force_scale=55.0,
  torque_scale=5.0,
  deadband=0.1,
  body_name=ROOT_TARGET,
)


def _observations() -> dict[str, ObservationGroupCfg]:
  def un(x: float) -> Unoise:
    return Unoise(n_min=-x, n_max=x)

  actor = {
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=un(0.05)
    ),
    "phase": ObservationTermCfg(func=reactor_mdp.phase, params=MOTION),
    "reference": ObservationTermCfg(func=reactor_mdp.reference_body_pos_b, params=MOTION),
    "base_height": ObservationTermCfg(func=reactor_mdp.root_height_observation),
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "previous_action": ObservationTermCfg(func=reactor_mdp.previous_action),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel, noise=un(0.2)),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel, noise=un(0.01)),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, noise=un(0.5)),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic = {
    "phase": ObservationTermCfg(func=reactor_mdp.phase, params=MOTION),
    "reference_progress": ObservationTermCfg(
      func=reactor_mdp.reference_phase, params=MOTION
    ),
    "reference_body_pos_b": ObservationTermCfg(
      func=reactor_mdp.reference_body_pos_b, params=MOTION
    ),
    "robot_body_pos_b": ObservationTermCfg(
      func=reactor_mdp.robot_body_pos_b, params=MOTION
    ),
    "wrench": ObservationTermCfg(func=reactor_mdp.wrench_action),
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
  def track(func, weight: float, std: float) -> RewardTermCfg:
    return RewardTermCfg(func=func, weight=weight, params={**MOTION, "std": std})

  return {
    # -- tracking (Table 1), root pair separate
    "root_position_xy": track(reactor_mdp.root_position_xy, 2.0, 0.3),
    "root_height": track(reactor_mdp.root_height, 10.0, 0.2),
    "root_orientation": track(reactor_mdp.root_orientation, 2.0, 0.4),
    "root_lin_vel": track(reactor_mdp.root_lin_vel, 0.5, 1.0),
    "root_ang_vel": track(reactor_mdp.root_ang_vel, 0.5, 3.14),
    # body terms carry the psi factor inside the function (Table 1)
    "body_position": track(reactor_mdp.body_position, 2.0, 0.3),
    "body_orientation": track(reactor_mdp.body_orientation, 2.0, 0.4),
    "survival": RewardTermCfg(func=envs_mdp.is_alive, weight=20.0),
    # -- regularization (Table 1). Action penalties cover the 20 controlled joints.
    "joint_torques_l2": RewardTermCfg(func=reactor_mdp.joint_torques_l2, weight=-1e-4),
    "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1e-6),
    "action_rate_l2": RewardTermCfg(func=reactor_mdp.joint_action_rate, weight=-1e-2),
    "action_jerk_l2": RewardTermCfg(func=reactor_mdp.joint_action_acceleration, weight=-1e-2),
    "wrench_force": RewardTermCfg(
      func=reactor_mdp.wrench_force_l1, weight=-1e-2, params={**MOTION}
    ),
    "wrench_torque": RewardTermCfg(
      func=reactor_mdp.wrench_torque_l1, weight=-1e-2, params={**MOTION}
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
    "scale_link_mass": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=("l_.*_link", "r_.*_link")
        ),
        "operation": "scale",
        "ranges": (0.9, 1.1),
      },
    ),
    "scale_actuator_gains": EventTermCfg(
      mode="startup",
      func=envs_mdp.dr.pd_gains,
      params={"kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1), "operation": "scale"},
    ),
  }
  if not play:
    events["push_robot"] = EventTermCfg(
      mode="interval",
      func=envs_mdp.push_by_setting_velocity,
      interval_range_s=(4.0, 8.0),
      params={
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.1, 0.1),
          "roll": (-0.26, 0.26),
          "pitch": (-0.26, 0.26),
          "yaw": (-0.39, 0.39),
        }
      },
    )
  return events


def _scene(envs: int) -> SceneCfg:
  return SceneCfg(
    num_envs=envs,
    env_spacing=2.5,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_pi_plus_robot_cfg()},
  )


def pi_plus_reactor_env_cfg(
  play: bool = False, motion_file: str | None = None, num_envs: int = 4096
) -> ManagerBasedRlEnvCfg:
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    episode_length_s=8.0,
    scene=_scene(1 if play else num_envs),
    observations=_observations(),
    actions={
      "joint_pos": envs_mdp.actions.JointPositionActionCfg(
        entity_name="robot",
        actuator_names=PI_PLUS_JOINT_ORDER,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
      ),
      "wrench": WRENCH_CFG,
    },
    commands={
      "motion": RetargetCommandCfg(
        entity_name="robot",
        debug_vis=not play,
        motion_files=(motion_file or str(MOTION_DIR),),
        source_bodies=SOURCE_BODIES,
        target_bodies=TARGET_BODIES,
        target_nominal=_target_nominal,
        psi_ramp_s=0.2,
        joint_position_range=(-0.15, 0.15),
      )
    },
    events=_events(play),
    rewards=_rewards(),
    terminations={
      "clip_end": TerminationTermCfg(func=reactor_mdp.clip_finished, params=MOTION, time_out=True),
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "root_pos": TerminationTermCfg(
        func=reactor_mdp.bad_root_position, params={**MOTION, "threshold": 1.0}
      ),
      "root_ori": TerminationTermCfg(
        func=reactor_mdp.bad_root_orientation, params={**MOTION, "threshold": 0.7854}
      ),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=ROOT_TARGET,
      distance=2.5,
      elevation=-5.0,
      azimuth=90.0,
    ),
    # Crouching clips need more constraint rows than the bare robot; 400 overflowed.
    sim=SimulationCfg(
      njmax=800,
      nconmax=256,
      mujoco=MujocoCfg(timestep=0.005, iterations=20, ls_iterations=30),
    ),
  )
  if play:
    cfg.episode_length_s = int(1e6)
    cfg.observations["actor"].enable_corruption = False
    motion = cfg.commands["motion"]
    assert isinstance(motion, RetargetCommandCfg)
    motion.sampling_mode = "uniform"
    motion.joint_position_range = (0.0, 0.0)
    motion.psi_ramp_s = 0.2
  return cfg