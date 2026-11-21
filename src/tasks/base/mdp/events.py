"""Startup events shared by the retargeting and tracking tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.mdp.actions import JointPositionAction

_ROBOT = SceneEntityCfg("robot")


def randomize_joint_default_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _ROBOT,
  pos_distribution_params: tuple[float, float] | None = None,
  operation: Literal["add", "scale", "abs"] = "abs",
  action_name: str = "joint_pos",
) -> None:
  """Perturb the nominal pose per env to mimic calibration error (startup).

  Keeps MuJoCo's ``qpos0`` untouched and only shifts what the policy treats as
  "default": ``default_joint_pos`` (observations, rewards) and the action offset.
  """
  if pos_distribution_params is None:
    return
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  joint_ids = torch.arange(default.shape[1], device=env.device)[asset_cfg.joint_ids]

  current = default[env_ids][:, joint_ids]
  sample = sample_uniform(*pos_distribution_params, current.shape, device=env.device)
  if operation == "add":
    current = current + sample
  elif operation == "scale":
    current = current * sample
  elif operation == "abs":
    current = sample
  else:
    raise ValueError(f"Unknown operation {operation!r}")
  default[env_ids[:, None], joint_ids[None, :]] = current

  term: JointPositionAction = env.action_manager.get_term(action_name)  # type: ignore[assignment]
  term._offset[env_ids] = default[env_ids][:, term._target_ids]


def randomize_default_joint_pos(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  offset_range: tuple[float, float] = (-0.01, 0.01),
  action_name: str = "joint_pos",
) -> None:
  """Add a uniform offset to every joint's nominal pose. AMP configs use this name."""
  randomize_joint_default_pos(
    env,
    env_ids,
    pos_distribution_params=offset_range,
    operation="add",
    action_name=action_name,
  )
