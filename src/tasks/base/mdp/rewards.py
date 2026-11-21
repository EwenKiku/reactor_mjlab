"""Regularizers shared by the retargeting and tracking tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.sensor import BuiltinSensor, ContactSensor

_ROBOT = SceneEntityCfg("robot")


def contact_history(env: ManagerBasedRlEnv, sensor_name: str, length: int | None = None):
  """[B, F] bool: contact force > 1 N in any of the last ``length`` substeps."""
  sensor: ContactSensor = env.scene[sensor_name]
  hist = sensor.data.force_history
  assert hist is not None, f"{sensor_name} needs history_length > 0"
  if length is not None:
    hist = hist[:, :, :length]
  return hist.norm(dim=-1).amax(dim=-1) > 1.0


def contact_index_for_bodies(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> list[int]:
  """Sensor columns ordered like ``asset_cfg`` bodies.

  Contact patterns are resolved in model order. A body config with
  ``preserve_order=True`` follows its name list, so the two sequences can
  disagree. Pi Plus ankles are right then left in the XML and left then
  right in ``FEET``.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.body_ids
  if isinstance(ids, slice):
    body_names = asset.body_names[ids]
  else:
    body_names = tuple(asset.body_names[i] for i in ids)
  primary = list(sensor.primary_names)
  return [primary.index(name) for name in body_names]


def energy(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(tau * vel), dim=-1)


def joint_torques_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  tau = env.scene[asset_cfg.name].data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(tau), dim=1)


def joint_rated_velocity_limit(
  env: ManagerBasedRlEnv, rated_velocity: tuple[float, ...], asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  vel = env.scene[asset_cfg.name].data.joint_vel
  rated = vel.new_tensor(rated_velocity)
  return torch.relu(vel.abs() / rated - 1.0).square().sum(dim=1)


def joint_rated_effort_limit(
  env: ManagerBasedRlEnv, rated_effort: tuple[float, ...], asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  tau = env.scene[asset_cfg.name].data.qfrc_actuator
  rated = tau.new_tensor(rated_effort)
  return torch.relu(tau.abs() / rated - 1.0).square().sum(dim=1)


class action_jerk_l2:
  """Second difference of actions; zero for the first two steps of an episode."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    am = env.action_manager
    jerk = (am.action - 2.0 * am.prev_action + am.prev_prev_action).square().sum(dim=-1)
    result = torch.where(self.steps >= 2, jerk, 0.0)
    self.steps += 1
    return result

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.steps[slice(None) if env_ids is None else env_ids] = 0


def _angular_momentum(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: BuiltinSensor = env.scene[sensor_name]
  return sensor.data


def _bounded(value: torch.Tensor, maximum: float) -> torch.Tensor:
  return torch.nan_to_num(value, nan=maximum, posinf=maximum, neginf=0.0).clamp(max=maximum)


def yaw_angular_momentum_l2(
  env: ManagerBasedRlEnv,
  sensor_name: str = "robot/root_angmom",
  command_name: str | None = None,
  yaw_deadband: float = 0.0,
) -> torch.Tensor:
  penalty = _bounded(_angular_momentum(env, sensor_name)[:, 2].square(), 1e3)
  if command_name is None or yaw_deadband <= 0.0:
    return penalty
  cmd = env.command_manager.get_command(command_name)
  turning = torch.abs(cmd[:, 2]) > yaw_deadband
  return torch.where(turning, torch.zeros_like(penalty), penalty)


class angular_momentum_rate_l2:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.previous = torch.zeros(env.num_envs, 3, device=env.device)
    self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(
    self, env: ManagerBasedRlEnv, sensor_name: str = "robot/root_angmom"
  ) -> torch.Tensor:
    current = _angular_momentum(env, sensor_name)
    finite = current.isfinite().all(dim=-1)
    penalty = _bounded(((current - self.previous) / env.step_dt).square().sum(dim=-1), 1e4)
    result = torch.where(self.valid & finite, penalty, 0.0)
    self.previous.copy_(torch.where(finite[:, None], current, self.previous))
    self.valid.copy_(finite)
    return result

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.valid[slice(None) if env_ids is None else env_ids] = False


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
  contact_history_length: int | None = 3,
) -> torch.Tensor:
  contacts = contact_history(env, sensor_name, contact_history_length)
  contacts = contacts[:, contact_index_for_bodies(env, sensor_name, asset_cfg)]
  asset: Entity = env.scene[asset_cfg.name]
  vel_xy = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]
  return torch.sum(vel_xy.norm(dim=-1) * contacts, dim=1)


def undesired_self_contacts(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Number of bodies with > 1 N self-contact force in the recent history."""
  return contact_history(env, sensor_name).float().sum(dim=1)
