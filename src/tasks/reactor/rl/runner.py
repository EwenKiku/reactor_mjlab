"""On-policy runner that performs the upper-level retargeting update.

The bilevel loop is the paper's single-loop TTSA variant, Eq. (3):

1. the environment rolls out the current reference ``g_t(p)`` and stores
   ``(frame, motion id, simulated state)`` samples;
2. the runner forms the simplified upper-level gradient of Eq. (6)-(7),
   ``dL = (1 - alpha) * dL/dg * dg/dp``, by back-propagating the tracking loss
   through the parameterized mapping with the simulated state detached;
3. ``p`` takes a projected gradient step and the buffer is cleared.

Everything else is the standard mjlab PPO loop.
"""

from __future__ import annotations

from src.tasks.reactor.mdp.retarget import NominalCalibration, RetargetMapping
from typing import TYPE_CHECKING, cast

import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.reactor.mdp.commands import RetargetCommand

if TYPE_CHECKING:
  from rsl_rl.env import VecEnv


class ReactorOnPolicyRunner(MjlabOnPolicyRunner):
  """PPO runner with the per-iteration retargeting parameter update."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self.upper_cfg = train_cfg.get("upper_level", {})
    self._upper_enabled = bool(self.upper_cfg.get("enabled", True))

  def _motion_command(self) -> RetargetCommand:
    return cast(RetargetCommand, self.env.unwrapped.command_manager.get_term("motion"))

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Wrap ``alg.update`` so every PPO iteration ends with one ``p`` step."""
    command = self._motion_command()
    original_update = self.alg.update

    def update_with_upper_level():
      loss_dict = original_update()
      if self._upper_enabled:
        batch = int(self.upper_cfg.get("batch_size", 0)) or None
        loss, grads = command.upper_level_gradient(batch)
        step = command.apply_upper_level_update(grads)
        command.last_upper_loss = loss
        writer = self.logger.writer
        if writer is not None:
          writer.add_scalar(
            "ReActor/upper_loss", loss, self.current_learning_iteration
          )
          writer.add_scalar(
            "ReActor/param_step", step, self.current_learning_iteration
          )
          for name, value in command.p.norms().items():
            writer.add_scalar(name, value, self.current_learning_iteration)
      return loss_dict

    self.alg.update = update_with_upper_level  # type: ignore[method-assign]
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len)
    finally:
      self.alg.update = original_update  # type: ignore[method-assign]

  def save(self, path: str, infos: dict | None = None) -> None:
    """Persist the retargeting parameters alongside the policy weights."""
    command = self._motion_command()
    extra = {
      "version": 2,
      "source_preprocessing": {
        "yaw": command.motion.source_yaw,
        "nominal_pos": command.motion.nominal_pos.cpu().numpy(),
        "nominal_quat": command.motion.nominal_quat.cpu().numpy(),
        "root_height": command.motion.source_height,
      },
      "retarget/p_pos": command.p.p_pos.detach().cpu(),
      "retarget/p_ori": command.p.p_ori.detach().cpu(),
      "retarget/p_z": command.p.p_z.detach().cpu(),
      "retarget/motion_names": list(command.motion.names),
      "retarget/motion_fingerprints": list(command.motion.fingerprints),
      "retarget/source_bodies": list(command.cfg.source_bodies),
      "retarget/target_bodies": list(command.cfg.target_bodies),
      "retarget/calibration": {
        "scale": command.calibration.scale,
        "x_nom": command.calibration.x_nom,
        "R_nom": command.calibration.R_nom,
        "source_height": command.calibration.source_height,
        "target_height": command.calibration.target_height,
      },
    }
    infos = {**(infos or {}), "reactor": extra}
    super().save(path, infos)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location or self.device)
    cmd = self._motion_command()
    state = infos["reactor"]
    if (tuple(state["retarget/source_bodies"]) != tuple(cmd.cfg.source_bodies) or
        tuple(state["retarget/target_bodies"]) != tuple(cmd.cfg.target_bodies)):
      raise ValueError("Checkpoint body correspondences do not match the environment")
    calib = NominalCalibration(**state["retarget/calibration"])
    cmd.calibration = calib
    cmd.mapping = RetargetMapping(calib, cmd.motion.z_nom * calib.scale, cmd.device)
    cmd.p.p_pos.copy_(state["retarget/p_pos"])
    cmd.p.p_ori.copy_(state["retarget/p_ori"])
    keys = state.get("retarget/motion_fingerprints")
    if keys is None:  # Earlier checkpoints only identified clips by name.
      keys, current_keys = state["retarget/motion_names"], cmd.motion.names
    else:
      current_keys = cmd.motion.fingerprints
    offsets = dict(zip(keys, state["retarget/p_z"].tolist(), strict=True))
    cmd.p.p_z.copy_(cmd.p.p_z.new_tensor([offsets.get(k, 0.) for k in current_keys]))
    cmd._cached_reference = None
    return infos

  def get_export_metadata(self, run_name: str) -> dict:
    metadata = super().get_export_metadata(run_name)
    command = self._motion_command()
    metadata["retarget_scale"] = float(command.calibration.scale)
    metadata["source_bodies"] = list(command.cfg.source_bodies)
    metadata["target_bodies"] = list(command.cfg.target_bodies)
    return metadata