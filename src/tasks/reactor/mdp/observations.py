from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from src.tasks.reactor.mdp.commands import RetargetCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> RetargetCommand:
  return cast(RetargetCommand, env.command_manager.get_term(command_name))


def previous_action(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.action_manager.prev_action


def root_height_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (env.scene["robot"].data.root_link_pos_w - env.scene.env_origins)[:, 2:3]


def clip_finished(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd = _command(env, command_name)
  return (cmd.psi >= 1) & (cmd.frames == cmd.motion.lengths[cmd.motion_ids] - 1)


def phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Retargeting phase variable ``psi`` in ``[0, 1]``."""
  return _command(env, command_name).psi.unsqueeze(-1)


def reference_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Normalized progress supplied to the critic."""
  command = _command(env, command_name)
  length = command.motion.lengths[command.motion_ids].clamp(min=1)
  return (command.frames.float() / length).unsqueeze(-1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Tracked body positions in the robot root frame."""
  from mjlab.utils.lab_api.math import quat_apply_inverse

  command = _command(env, command_name)
  root_pos = command.robot.data.root_link_pos_w
  root_quat = command.robot.data.root_link_quat_w
  num_bodies = command.robot_body_pos_w.shape[1]
  rel = quat_apply_inverse(
    root_quat[:, None, :].expand(-1, num_bodies, -1),
    command.robot_body_pos_w - root_pos[:, None, :],
  )
  return rel.flatten(1)


def reference_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference body positions in the robot root frame."""
  return _command(env, command_name).command


def wrench_action(env: ManagerBasedRlEnv, action_name: str = "wrench") -> torch.Tensor:
  return env.action_manager.get_term(action_name).applied_wrench.flatten(1)


##
# Terminations (paper: root deviates by > 1 m or > 45 deg).
##


def bad_root_position(
  env: ManagerBasedRlEnv, command_name: str, threshold: float = 1.0
) -> torch.Tensor:
  command = _command(env, command_name)
  reference = command.reference().pos[:, 0] + env.scene.env_origins
  error = torch.norm(reference - env.scene["robot"].data.root_link_pos_w, dim=-1)
  return error > threshold


def bad_root_orientation(
  env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.7854
) -> torch.Tensor:
  """Terminate when the root geodesic error exceeds ``threshold`` radians."""
  from mjlab.utils.lab_api.math import quat_error_magnitude

  command = _command(env, command_name)
  error = quat_error_magnitude(
    command.reference().quat[:, 0], env.scene["robot"].data.root_link_quat_w
  )
  return error > threshold


def bad_body_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float = 0.5,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Terminate when any tracked end-effector drifts too far from its reference."""
  command = _command(env, command_name)
  reference = command.reference().pos + env.scene.env_origins[:, None, :]
  error = torch.norm(reference - command.robot_body_pos_w, dim=-1)
  if body_names is not None:
    ids = [i for i, name in enumerate(command.cfg.target_bodies) if name in body_names]
    error = error[:, ids]
  return torch.any(error > threshold, dim=-1)