"""OrderedJointPositionAction"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(JointPositionActionCfg):
    """Joint targets in explicit ``joint_names`` order (matches Evo1 AMP flat)."""
    joint_names: tuple[str, ...]
    preserve_order: bool = True
    def build(self, env: ManagerBasedRlEnv) -> OrderedJointPositionAction:
        return OrderedJointPositionAction(self, env)


class OrderedJointPositionAction(JointPositionAction):
    """``find_joints(joint_names)`` + clamp raw actions before scale/offset"""

    cfg: OrderedJointPositionActionCfg
    _clip_raw: float = 100.0

    def _find_targets(self, cfg: OrderedJointPositionActionCfg):
        return self._entity.find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(
            torch.clamp(actions, -self._clip_raw, self._clip_raw)
        )
