"""ReActor tracking rewards (paper Table 1).

The reward is ``r = r_tracking + r_regularization``.  Tracking terms are summed
over the rigid-body pairs, with the root pair treated separately.  Terms that
depend on proper tracking are scaled by the retargeting phase ``psi`` so the
policy is not punished while it moves into the start pose; trajectory segments
with ``psi < 1`` are excluded from the upper-level buffer (see ``commands``).

Tracking terms use the negative squared losses of Table 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from src.tasks.reactor.mdp.commands import RetargetCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def _command(env: ManagerBasedRlEnv, command_name: str) -> RetargetCommand:
  return cast(RetargetCommand, env.command_manager.get_term(command_name))


def _tracking(
  env: ManagerBasedRlEnv, command_name: str, term: str, std: float
) -> torch.Tensor:
  command = _command(env, command_name)
  reference = command.reference()
  origins = env.scene.env_origins[:, None, :]
  if term == "root_pos_xy":
    error = torch.sum(
      ((reference.pos[:, 0] + origins[:, 0]) - env.scene["robot"].data.root_link_pos_w)[..., :2] ** 2,
      dim=-1,
    )
  elif term == "root_height":
    error = (
      (reference.pos[:, 0, 2] + origins[:, 0, 2])
      - env.scene["robot"].data.root_link_pos_w[:, 2]
    ) ** 2
  elif term == "root_ori":
    error = quat_error_magnitude(
      reference.quat[:, 0], env.scene["robot"].data.root_link_quat_w
    ) ** 2
  elif term == "root_lin_vel":
    error = torch.sum(
      (reference.lin_vel[:, 0] - command.robot_body_lin_vel_w[:, 0]) ** 2, dim=-1
    )
  elif term == "root_ang_vel":
    error = torch.sum(
      (reference.ang_vel[:, 0] - command.robot_body_ang_vel_w[:, 0]) ** 2, dim=-1
    )
  elif term == "body_pos":
    error = torch.sum(
      ((reference.pos + origins) - command.robot_body_pos_w)[:, 1:] ** 2, dim=-1
    )
  elif term == "body_ori":
    error = quat_error_magnitude(
      reference.quat[:, 1:], command.robot_body_quat_w[:, 1:]
    ).square()
  else:
    raise ValueError(f"Unknown tracking term {term!r}")
  return -error.sum(-1) if error.ndim == 2 else -error


def root_position_xy(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.3
) -> torch.Tensor:
  return _tracking(env, command_name, "root_pos_xy", std)


def root_height(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.3
) -> torch.Tensor:
  return _tracking(env, command_name, "root_height", std)


def root_orientation(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.4
) -> torch.Tensor:
  return _tracking(env, command_name, "root_ori", std)


def root_lin_vel(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.6
) -> torch.Tensor:
  return _tracking(env, command_name, "root_lin_vel", std)


def root_ang_vel(
  env: ManagerBasedRlEnv, command_name: str, std: float = 1.0
) -> torch.Tensor:
  return _tracking(env, command_name, "root_ang_vel", std)


def body_position(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.3
) -> torch.Tensor:
  """Per-body position reward, scaled by ``psi`` (Table 1: weight x psi)."""
  command = _command(env, command_name)
  return _tracking(env, command_name, "body_pos", std) * command.psi


def body_orientation(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.4
) -> torch.Tensor:
  command = _command(env, command_name)
  return _tracking(env, command_name, "body_ori", std) * command.psi


##
# Regularization.
##


def wrench_force_l1(
  env: ManagerBasedRlEnv, command_name: str = "motion", action_name: str = "wrench"
) -> torch.Tensor:
  """``|f|_1`` on the residual force, scaled by ``psi`` (Table 1)."""
  term = env.action_manager.get_term(action_name)
  psi = _command(env, command_name).psi
  return term.applied_wrench[:, :3].abs().sum(dim=-1) * psi


def wrench_torque_l1(
  env: ManagerBasedRlEnv, command_name: str = "motion", action_name: str = "wrench"
) -> torch.Tensor:
  """``|tau|_1`` on the residual torque, scaled by ``psi`` (Table 1)."""
  term = env.action_manager.get_term(action_name)
  psi = _command(env, command_name).psi
  return term.applied_wrench[:, 3:].abs().sum(dim=-1) * psi


def joint_torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  tau = env.scene[asset_cfg.name].data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(tau), dim=1)


def joint_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  am = env.action_manager
  joint = slice(0, am.get_term("joint_pos").action_dim)
  return (am.action[:, joint] - am.prev_action[:, joint]).square().sum(-1)


def joint_action_acceleration(env: ManagerBasedRlEnv) -> torch.Tensor:
  am = env.action_manager
  joint = slice(0, am.get_term("joint_pos").action_dim)
  delta = am.action[:, joint] - 2 * am.prev_action[:, joint] + am.prev_prev_action[:, joint]
  return delta.square().sum(-1) * (env.episode_length_buf > 2)
