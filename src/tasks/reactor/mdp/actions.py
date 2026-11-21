"""Root residual force/torque action (paper Eq. 18-19, residual force control).

The policy outputs joint position setpoints *plus* an auxiliary wrench applied
directly to the root body.  A continuous deadband makes it easy for the policy
to predict exactly zero, and the wrench is penalised in the reward so it stays
a last resort (handstands and other motions a humanoid cannot reproduce
honestly).

The policy specifies a body-frame wrench, transformed to world axes when
writing MuJoCo xfrc_applied.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from src.tasks.reactor.mdp.retarget import deadband

from mjlab.envs import ManagerBasedRlEnv

if TYPE_CHECKING:
  pass


@dataclasses.dataclass(kw_only=True)
class RootWrenchActionCfg(ActionTermCfg):
  """Six-dimensional wrench on the root body."""

  force_scale: float = 200.0
  """Newton per unit of raw action, per world axis."""
  torque_scale: float = 50.0
  """Newton-metre per unit of raw action, per world axis."""
  deadband: float = 0.1
  """Deadband applied to the *scaled* wrench (the paper's ``d``)."""
  body_name: str = ""
  """Body the wrench is applied to.  Defaults to the entity's root body."""

  def build(self, env: ManagerBasedRlEnv) -> RootWrenchAction:
    return RootWrenchAction(self, env)


class RootWrenchAction(ActionTerm):
  """Scale a local wrench, apply its deadband, then rotate it to world axes."""

  cfg: RootWrenchActionCfg

  def __init__(self, cfg: RootWrenchActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)
    self._applied_wrench = torch.zeros_like(self._raw_actions)
    scale = torch.tensor(
      [cfg.force_scale] * 3 + [cfg.torque_scale] * 3, device=self.device
    )
    self._scale = scale.unsqueeze(0).repeat(self.num_envs, 1)
    body_name = cfg.body_name or self._entity.body_names[0]
    self._body_index = self._entity.body_names.index(body_name)
    self._name = body_name

  @property
  def name(self) -> str:
    return self._name

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def applied_wrench(self) -> torch.Tensor:
    """Applied force and torque in root axes, after the deadband."""
    return self._applied_wrench

  @property
  def raw_force_norm(self) -> torch.Tensor:
    return self._raw_actions[:, :3].norm(dim=-1)

  @property
  def raw_torque_norm(self) -> torch.Tensor:
    return self._raw_actions[:, 3:].norm(dim=-1)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale
    self._applied_wrench = deadband(self._processed_actions, self.cfg.deadband)

  def apply_actions(self) -> None:
    root_quat = self._entity.data.root_link_quat_w
    self._entity.write_external_wrench_to_sim(
      quat_apply(root_quat, self._applied_wrench[:, :3])[:, None],
      quat_apply(root_quat, self._applied_wrench[:, 3:])[:, None],
      body_ids=[self._body_index],
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._processed_actions[env_ids] = 0.0
    self._applied_wrench[env_ids] = 0.0
    count = self.num_envs if isinstance(env_ids, slice) else env_ids.numel()
    if count:
      zeros = torch.zeros(count, 1, 3, device=self.device)
      self._entity.write_external_wrench_to_sim(
        zeros, zeros, env_ids=env_ids, body_ids=[self._body_index]
      )