"""Observation and reward terms for the downstream tracking task (paper Table 6).

The upstream ``mjlab.tasks.tracking`` terms treat the anchor as one lump: position
and orientation only.  Table 6 scores *five* root terms separately (xy, height,
orientation, linear velocity, angular velocity) and weights the tracked bodies
independently, so the split lives here next to the rest of the ReActor port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


##
# Observations.
##


def reference_joint_target(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference joint positions minus the robot's (deployable tracking command)."""
  command = _command(env, command_name)
  return command.joint_pos - command.robot_joint_pos


def reference_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Normalized progress through the current clip, in ``[0, 1]``."""
  command = _command(env, command_name)
  total = max(int(command.motion.time_step_total), 1)
  return (command.time_steps.float() / total).unsqueeze(-1)


##
# Table 6 tracking rewards.
##


def _body_mask(command: MotionCommand, body_names: tuple[str, ...] | None):
  if body_names is None:
    return slice(None)
  ids = [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
  return ids


def root_position_xy(
  env: ManagerBasedRlEnv, command_name: str, std: float, body_names=None
) -> torch.Tensor:
  del body_names
  command = _command(env, command_name)
  error = torch.sum(
    torch.square((command.anchor_pos_w - command.robot_anchor_pos_w)[..., :2]), dim=-1
  )
  return torch.exp(-error / std**2)


def root_height(
  env: ManagerBasedRlEnv, command_name: str, std: float, body_names=None
) -> torch.Tensor:
  del body_names
  command = _command(env, command_name)
  error = (command.anchor_pos_w[..., 2] - command.robot_anchor_pos_w[..., 2]) ** 2
  return torch.exp(-error / std**2)


def root_orientation(
  env: ManagerBasedRlEnv, command_name: str, std: float, body_names=None
) -> torch.Tensor:
  del body_names
  command = _command(env, command_name)
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def root_lin_vel(
  env: ManagerBasedRlEnv, command_name: str, std: float, body_names=None
) -> torch.Tensor:
  del body_names
  command = _command(env, command_name)
  error = torch.sum(
    torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w), dim=-1
  )
  return torch.exp(-error / std**2)


def root_ang_vel(
  env: ManagerBasedRlEnv, command_name: str, std: float, body_names=None
) -> torch.Tensor:
  del body_names
  command = _command(env, command_name)
  error = torch.sum(
    torch.square(command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w), dim=-1
  )
  return torch.exp(-error / std**2)


def body_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _command(env, command_name)
  ids = _body_mask(command, body_names)
  error = torch.sum(
    torch.square(command.body_pos_relative_w[:, ids] - command.robot_body_pos_w[:, ids]),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def body_orientation(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = _command(env, command_name)
  ids = _body_mask(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, ids], command.robot_body_quat_w[:, ids]
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)