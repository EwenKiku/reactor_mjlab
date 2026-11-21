"""ReActor retargeting command: source clip playback + the parameterized reference.

The term owns three things:

* **Source clips** -- SOMA body trajectories converted by
  ``tools/convert_soma_bvh_to_npz.py``.  Each environment tracks one clip and
  advances its own frame index at the control rate.
* **Retargeting parameters** ``p`` -- ``p_pos``/``p_ori`` per correspondence and
  a per-clip ``p_z``.  They are differentiable and are updated by the runner
  through :meth:`apply_upper_level_update`.
* **The parameterized reference** ``g_t`` -- Eq. (10)-(13) applied to the current
  source frame.  Rewards and observations read it; the upper level compares it
  to the simulated state.

Rollout buffers feed the upper level: every step stores the frame index, the
motion id, ``psi`` and the simulated body frames.  The runner samples from that
buffer each iteration and takes one TTSA step on ``p`` (Eq. 3, 6, 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  sample_uniform,
)

from src.tasks.reactor.mdp.retarget import (
  LossWeights,
  NominalCalibration,
  RetargetedState,
  RetargetMapping,
  RetargetParams,
  retargeting_loss,
)

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv


_QUAT_TO_WXYZ = {"wxyz": (0, 1, 2, 3), "xyzw": (3, 0, 1, 2)}


def _expand_motion_paths(paths: list[str] | tuple[str, ...]) -> list[str]:
  """Files stay as given; a directory contributes every ``*.npz`` inside it."""
  expanded: list[str] = []
  for raw in paths:
    path = Path(raw).expanduser()
    if path.is_dir():
      found = sorted(path.glob("*.npz"))
      if not found:
        raise FileNotFoundError(f"No .npz files in {path}")
      expanded.extend(str(item) for item in found)
    else:
      expanded.append(str(path))
  if not expanded:
    raise ValueError("No source motion files were configured.")
  return expanded


class SourceMotionLoader:
  """Source body trajectories for every clip, concatenated along time.

  ``starts``/``lengths`` delimit each clip inside the concatenated tensors, so a
  single ``(motion_id, frame)`` pair indexes the whole library.
  """

  def __init__(
    self,
    motion_files: list[str] | tuple[str, ...],
    source_bodies: tuple[str, ...],
    device: str | torch.device = "cpu",
    quaternion_order: str = "wxyz",
    motion_weights: dict[str, float] | None = None,
    clamp_feet: bool = True,
  ) -> None:
    paths = _expand_motion_paths(motion_files)
    clips = [
      self._load(path, source_bodies, device, quaternion_order) for path in paths
    ]
    fps_set = {clip["fps"] for clip in clips if clip["fps"] is not None}
    if len(fps_set) > 1:
      raise ValueError(f"Source clips have different frame rates: {sorted(fps_set)}")
    self.fps = next(iter(fps_set)) if fps_set else None
    self.names = tuple(Path(p).stem for p in paths)
    if len(set(self.names)) != len(self.names):
      raise ValueError("Source motion names must be unique")
    self.fingerprints = tuple(c["fingerprint"] for c in clips)
    self.num_motions = len(clips)
    self.lengths = torch.tensor(
      [c["pos"].shape[0] for c in clips], dtype=torch.long, device=device
    )
    self.starts = torch.zeros(self.num_motions, dtype=torch.long, device=device)
    if self.num_motions > 1:
      self.starts[1:] = torch.cumsum(self.lengths[:-1], dim=0)
    self.pos = torch.cat([c["pos"] for c in clips])
    self.quat = torch.cat([c["quat"] for c in clips])
    self.lin_vel = torch.cat([c["lin_vel"] for c in clips])
    self.ang_vel = torch.cat([c["ang_vel"] for c in clips])
    self.z_nom = torch.tensor(
      [c["z_nom"] for c in clips], dtype=torch.float32, device=device
    )
    # Nominal calibration constants come from the clip metadata; every clip of
    # the same skeleton carries the same values.
    first = clips[0]
    self.nominal_pos = torch.as_tensor(first["nominal_pos"], device=device)
    self.nominal_quat = torch.as_tensor(first["nominal_quat"], device=device)
    self.source_height = float(first["root_height"])
    self.source_yaw = first["source_yaw"]
    for clip in clips[1:]:
      if not (np.allclose(clip["nominal_pos"], first["nominal_pos"]) and
              np.allclose(clip["nominal_quat"], first["nominal_quat"]) and
              clip["source_yaw"] == self.source_yaw):
        raise ValueError("Source clips use inconsistent nominal calibrations")
    self.time_step_total = int(self.lengths.max())

    weights = torch.tensor(
      [(motion_weights or {}).get(name, 1.0) for name in self.names],
      dtype=torch.float32,
      device=device,
    )
    if torch.any(weights < 0) or float(weights.sum()) <= 0:
      raise ValueError(f"motion_weights must be non-negative and sum > 0: {weights}")
    self.sample_prob = weights / weights.sum()

  @staticmethod
  def _load(
    path: str,
    source_bodies: tuple[str, ...],
    device: str | torch.device,
    quaternion_order: str,
  ) -> dict:
    data = np.load(path)
    order = quaternion_order
    if "quaternion_order" in data.files:
      order = str(np.asarray(data["quaternion_order"]).item())
    if order not in _QUAT_TO_WXYZ:
      raise ValueError(f"Unsupported quaternion order {order!r} in {path}")

    src_names = [str(n) for n in data["body_names"]]
    missing = sorted(set(source_bodies) - set(src_names))
    if missing:
      raise ValueError(f"{path} is missing source bodies {missing}")
    idx = [src_names.index(name) for name in source_bodies]

    def load(key: str) -> torch.Tensor:
      return torch.as_tensor(data[key], dtype=torch.float32, device=device)

    # A renamed clip retains its learned floor correction. Two different
    # performances with the same basename must never share that correction.
    pos = np.ascontiguousarray(data["body_pos_w"][:, idx], dtype=np.float32)
    quat = np.ascontiguousarray(data["body_quat_w"][:, idx][..., list(_QUAT_TO_WXYZ[order])], dtype=np.float32)
    digest = hashlib.sha256(pos)
    digest.update(quat)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else None
    return {
      "fps": fps,
      "source_yaw": float(data["source_yaw"]) if "source_yaw" in data else None,
      "fingerprint": digest.hexdigest(),
      "pos": torch.as_tensor(pos, device=device),
      "quat": torch.as_tensor(quat, device=device),
      "lin_vel": load("body_lin_vel_w")[:, idx],
      "ang_vel": load("body_ang_vel_w")[:, idx],
      "z_nom": float(np.asarray(data["z_nom"]).reshape(-1)[0]),
      "nominal_pos": np.asarray(data["nominal_pos"], dtype=np.float32)[idx],
      "nominal_quat": np.asarray(data["nominal_quat"], dtype=np.float32)[idx][
        ..., list(_QUAT_TO_WXYZ[order])
      ],
      "root_height": float(np.asarray(data["root_height"]).reshape(-1)[0]),
    }

  def frame_index(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    return self.starts[motion_ids] + frames


class RetargetCommand(CommandTerm):
  """Holds the source clips, ``p`` and the per-step reference ``g_t``."""

  cfg: RetargetCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: RetargetCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.motion = SourceMotionLoader(
      cfg.motion_files,
      tuple(cfg.source_bodies),
      device=self.device,
      quaternion_order=cfg.quaternion_order,
      motion_weights=cfg.motion_weights,
    )
    control_fps = 1.0 / env.step_dt
    if self.motion.fps is not None and abs(self.motion.fps - control_fps) > 1e-3:
      raise ValueError(
        f"Source clips are {self.motion.fps} fps but the policy runs at "
        f"{control_fps} Hz; resample them first."
      )

    if cfg.target_nominal is None:
      raise ValueError("RetargetCommandCfg.target_nominal must be provided.")
    target_pos, target_quat, target_height = cfg.target_nominal(env)
    self.calibration = NominalCalibration.build(
      source_pos=self.motion.nominal_pos.cpu().numpy().astype(np.float64),
      source_quat=self.motion.nominal_quat.cpu().numpy().astype(np.float64),
      target_pos=target_pos,
      target_quat=target_quat,
      source_height=self.motion.source_height,
      target_height=target_height,
    )
    # ``z_nom`` is measured on the unscaled source, but Eq. (10) applies the
    # vertical shift after the global scale, so convert it to target units.
    self.mapping = RetargetMapping(
      self.calibration,
      self.motion.z_nom * self.calibration.scale,
      self.device,
    )

    # Learnable parameters, one row per correspondence (+ one p_z per clip).
    self.p = RetargetParams(
      p_pos=torch.zeros(len(cfg.source_bodies), 3, device=self.device),
      p_ori=torch.zeros(len(cfg.source_bodies), 3, device=self.device),
      p_z=torch.zeros(self.motion.num_motions, device=self.device),
    ).detach()
    self._requires_grad(False)

    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.target_bodies, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.failed_count = torch.zeros(self.motion.num_motions, device=self.device)
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Retargeting phase variable psi in [0, 1]; 1 means "tracking active".
    self.psi = torch.zeros(self.num_envs, device=self.device)

    self.upper_buffer = UpperLevelBuffer(
      capacity=cfg.upper_buffer_size,
      num_bodies=len(cfg.source_bodies),
      device=self.device,
    )
    self.loss_weights = LossWeights(
      pos=cfg.loss_pos_weight,
      ori=cfg.loss_ori_weight,
      lin_vel=cfg.loss_lin_vel_weight,
      ang_vel=cfg.loss_ang_vel_weight,
    )
    self._cached_reference = None
    self.upper_level_steps = 0
    self.last_upper_loss = 0.0
    self._last_param_step = 0.0

    for name in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_body_pos",
      "error_body_rot",
      "upper_loss",
      "param_step",
      "psi",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
    print(
      "[INFO] ReActor source clips: "
      + ", ".join(
        f"{name} ({int(length)} frames)"
        for name, length in zip(self.motion.names, self.motion.lengths.tolist(), strict=True)
      )
    )

  def _requires_grad(self, flag: bool) -> None:
    for tensor in self.p.as_list():
      tensor.requires_grad_(flag)

  ##
  # Reference.
  ##

  def _source_slice(self, env_ids: torch.Tensor | slice) -> torch.Tensor:
    ids = self.motion_ids[env_ids]
    fr = self.frames[env_ids]
    return self.motion.frame_index(ids, fr)

  def _source_frames(self, env_ids: torch.Tensor | slice) -> tuple[torch.Tensor, ...]:
    index = self._source_slice(env_ids)
    return (
      self.motion.pos[index],
      self.motion.quat[index],
      self.motion.lin_vel[index],
      self.motion.ang_vel[index],
    )

  def reference(self, env_ids: torch.Tensor | slice = slice(None)) -> RetargetedState:
    """Parameterized reference ``g_t`` for the current frame of every env."""
    full = isinstance(env_ids, slice) and env_ids == slice(None)
    if full and self._cached_reference is not None:
      return self._cached_reference
    ids = self.motion_ids[env_ids]
    pos, quat, lin_vel, ang_vel = self._source_frames(env_ids)
    params = self.p
    if not self.p.p_pos.requires_grad:
      params = params.detach()
    result = self.mapping(
      params,
      pos,
      quat,
      lin_vel,
      ang_vel,
      self.motion.z_nom[ids] * self.calibration.scale + params.p_z[ids],
      self.psi[env_ids],
    )

    if full:
      self._cached_reference = result
    return result

  ##
  # Reference views used by rewards / observations.
  ##

  def _ref_slice(self, env_ids: torch.Tensor | slice) -> RetargetedState:
    return self.reference(env_ids)

  @property
  def ref_pos(self) -> torch.Tensor:
    return self.reference().pos + self._env.scene.env_origins[:, None, :]

  @property
  def ref_quat(self) -> torch.Tensor:
    return self.reference().quat

  @property
  def ref_lin_vel(self) -> torch.Tensor:
    return self.reference().lin_vel

  @property
  def ref_ang_vel(self) -> torch.Tensor:
    return self.reference().ang_vel

  @property
  def root_index(self) -> int:
    return 0

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def command(self) -> torch.Tensor:
    """Reference body poses expressed in the root frame (policy input)."""
    ref = self.reference()
    robot_root_pos = self.robot.data.root_link_pos_w
    robot_root_quat = self.robot.data.root_link_quat_w
    rel_pos, rel_quat = self._in_root_frame(
      ref.pos + self._env.scene.env_origins[:, None, :], ref.quat, robot_root_pos, robot_root_quat
    )
    root = robot_root_quat[:, None].expand_as(ref.quat)
    return torch.cat([rel_pos.flatten(1), _rot6d(rel_quat),
                      quat_apply_inverse(root, ref.lin_vel).flatten(1),
                      quat_apply_inverse(root, ref.ang_vel).flatten(1)], dim=-1)

  @staticmethod
  def _in_root_frame(
    pos_w: torch.Tensor, quat_w: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    n = pos_w.shape[0]
    rel_pos = quat_apply_inverse(root_quat.unsqueeze(1).expand(n, pos_w.shape[1], 4), pos_w - root_pos.unsqueeze(1))
    rel_quat = quat_mul(quat_inv(root_quat.unsqueeze(1).expand_as(quat_w)), quat_w)
    return rel_pos, rel_quat

  ##
  # CommandTerm API.
  ##

  def _update_metrics(self) -> None:
    ref = self.reference()
    origins = self._env.scene.env_origins[:, None, :]
    self.metrics["error_anchor_pos"] = torch.norm(
      (ref.pos[:, self.root_index] + origins[:, 0]) - self.robot.data.root_link_pos_w, dim=-1
    )
    from mjlab.utils.lab_api.math import quat_error_magnitude

    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      ref.quat[:, self.root_index], self.robot.data.root_link_quat_w
    )
    self.metrics["error_body_pos"] = torch.norm(
      (ref.pos + origins) - self.robot_body_pos_w, dim=-1
    ).mean(-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      ref.quat, self.robot_body_quat_w
    ).mean(-1)
    self.metrics["upper_loss"].fill_(self.last_upper_loss)
    self.metrics["param_step"].fill_(self._last_param_step)
    self.metrics["psi"] = self.psi

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    if self.cfg.sampling_mode == "adaptive":
      self._record_failures(env_ids)
    self._cached_reference = None
    self._sample_clip_and_frame(env_ids)
    self.psi[env_ids] = 0.0 if self.cfg.psi_ramp_s > 0 else 1.0
    self._write_root_state_to_sim(env_ids)

  def _sample_clip_and_frame(self, env_ids: torch.Tensor) -> None:
    if self.cfg.sampling_mode == "start":
      self.motion_ids[env_ids] = env_ids % self.motion.num_motions
      self.frames[env_ids] = 0
      return
    if self.cfg.sampling_mode == "adaptive" and self.failed_count.sum() > 0:
      mix = self.cfg.adaptive_uniform_ratio
      probs = (1 - mix) * self.failed_count / self.failed_count.sum() + mix * self.motion.sample_prob
    else:
      probs = self.motion.sample_prob
    self.motion_ids[env_ids] = torch.multinomial(
      probs, env_ids.numel(), replacement=True
    )
    lengths = self.motion.lengths[self.motion_ids[env_ids]]
    self.frames[env_ids] = (
      torch.rand(env_ids.numel(), device=self.device) * lengths.float()
    ).long().clamp(max=lengths - 1)

  def _write_root_state_to_sim(self, env_ids: torch.Tensor) -> None:
    """Initialize the root from the mapped source and perturb the nominal joints."""
    ref = self.reference(env_ids)
    root_pos = ref.pos[:, self.root_index] + self._env.scene.env_origins[env_ids]
    root_quat = ref.quat[:, self.root_index]
    root_lin_vel = ref.lin_vel[:, self.root_index]
    root_ang_vel = ref.ang_vel[:, self.root_index]

    default_joint_pos = self.robot.data.default_joint_pos[env_ids]
    low, high = self.cfg.joint_position_range
    joint_pos = default_joint_pos + sample_uniform(
      low, high, default_joint_pos.shape, device=self.device
    )
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft[..., 0], soft[..., 1])
    joint_vel = torch.zeros_like(joint_pos)

    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)
    self._env.sim.forward()

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if env_ids is not None:
      return  # explicit reset computes observations at the sampled frame
    self._cached_reference = None
    if self.cfg.collect_upper:
      self._record_upper_level()
    active = self.psi >= 1.0
    self.frames[:] = torch.minimum(self.frames + active.long(), self.motion.lengths[self.motion_ids] - 1)
    if self.cfg.psi_ramp_s > 0:
      self.psi.add_(self._env.step_dt / self.cfg.psi_ramp_s).clamp_(max=1.0)

  ##
  # Upper level.
  ##

  def _record_failures(self, env_ids: torch.Tensor) -> None:
    terminated = self._env.termination_manager.terminated[env_ids]
    failed = env_ids[terminated]
    if failed.numel() == 0:
      return
    counts = torch.bincount(self.motion_ids[failed], minlength=self.motion.num_motions)
    self.failed_count += counts.float()

  def _record_upper_level(self) -> None:
    """Store one rollout sample per env (only once tracking is active)."""
    active = self.psi >= 1.0
    if not torch.any(active):
      return
    self.upper_buffer.push(
      motion_ids=self.motion_ids[active],
      frames=self.frames[active],
      z_nom=self.motion.z_nom[self.motion_ids[active]],
      sim_pos=self.robot_body_pos_w[active] - self._env.scene.env_origins[active, None, :],
      sim_quat=self.robot_body_quat_w[active],
      sim_lin_vel=self.robot_body_lin_vel_w[active],
      sim_ang_vel=self.robot_body_ang_vel_w[active],
    )

  def upper_level_gradient(
    self, batch_size: int | None = None
  ) -> tuple[float, list[torch.Tensor | None]]:
    """One TTSA gradient estimate over the rollout buffer (Eq. 6-7).

    Returns ``(loss, grad_norm)``.  Gradients accumulate into ``self.p``.
    """
    if self.upper_buffer.size == 0:
      return 0.0, [None] * len(self.p.as_list())
    sample = self.upper_buffer.sample(
      batch_size or self.upper_buffer.capacity, self.device
    )
    self._requires_grad(True)
    try:
      reference = self._reference_for(
        sample["motion_ids"].long(), sample["frames"].long(), sample["z_nom"]
      )
      loss, _ = retargeting_loss(
        reference=reference,
        simulated_pos=sample["sim_pos"],
        simulated_quat=sample["sim_quat"],
        simulated_lin_vel=sample["sim_lin_vel"],
        simulated_ang_vel=sample["sim_ang_vel"],
        weights=self.loss_weights,
        root_index=self.root_index,
      )
      total = loss.mean()
      grads = list(
        torch.autograd.grad(total, self.p.as_list(), allow_unused=True)
      )
    finally:
      self._requires_grad(False)
    return float(total.detach()), grads

  def _reference_for(
    self, motion_ids: torch.Tensor, frames: torch.Tensor, z_nom: torch.Tensor
  ) -> RetargetedState:
    index = self.motion.frame_index(motion_ids, frames)
    return self.mapping(
      self.p,
      self.motion.pos[index],
      self.motion.quat[index],
      self.motion.lin_vel[index],
      self.motion.ang_vel[index],
      z_nom * self.calibration.scale + self.p.p_z[motion_ids],
      torch.ones_like(z_nom),
    )

  def apply_upper_level_update(self, grads: list[torch.Tensor | None]) -> float:
    """Projected gradient step ``p <- Proj_P(p - eta * dL)`` (Eq. 3)."""
    eta = self.cfg.param_lr
    scale = (1.0 - self.cfg.alpha) * eta
    step_norm = 0.0
    with torch.no_grad():
      for tensor, grad in zip(self.p.as_list(), grads, strict=True):
        if grad is None:
          continue
        delta = grad * scale
        step_norm = max(step_norm, float(delta.norm()))
        tensor -= delta
    RetargetParams.project_(
      self.p, self.cfg.delta_pos, self.cfg.delta_ori, self.cfg.delta_z
    )
    self._cached_reference = None
    self.upper_buffer.clear()
    self.upper_level_steps += 1
    self._last_param_step = step_norm
    return step_norm


def _rot6d(quat: torch.Tensor) -> torch.Tensor:
  """First two rotation-matrix columns, flattened over the body axis."""
  from mjlab.utils.lab_api.math import matrix_from_quat

  mat = matrix_from_quat(quat)
  return mat[..., :2].reshape(*quat.shape[:-2], -1)


class UpperLevelBuffer:
  """Rollout samples ``(frame, motion id, simulated body frames)``."""

  def __init__(self, capacity: int, num_bodies: int, device: str | torch.device):
    self.capacity = capacity
    self.num_bodies = num_bodies
    self.device = device
    self.motion_ids = torch.zeros(capacity, dtype=torch.long, device=device)
    self.frames = torch.zeros(capacity, dtype=torch.long, device=device)
    self.sim_pos = torch.zeros(capacity, num_bodies, 3, device=device)
    self.sim_quat = torch.zeros(capacity, num_bodies, 4, device=device)
    self.sim_lin_vel = torch.zeros(capacity, num_bodies, 3, device=device)
    self.sim_ang_vel = torch.zeros(capacity, num_bodies, 3, device=device)
    self.z_nom = torch.zeros(capacity, device=device)
    self.size = 0
    self._cursor = 0

  def push(
    self, motion_ids, frames, z_nom, sim_pos, sim_quat, sim_lin_vel, sim_ang_vel
  ) -> None:
    n = motion_ids.numel()
    if n > self.capacity:
      keep = torch.randperm(n, device=self.device)[: self.capacity]
      motion_ids, frames, z_nom = motion_ids[keep], frames[keep], z_nom[keep]
      sim_pos, sim_quat = sim_pos[keep], sim_quat[keep]
      sim_lin_vel, sim_ang_vel = sim_lin_vel[keep], sim_ang_vel[keep]
      n = self.capacity
    index = (self._cursor + torch.arange(n, device=self.device)) % self.capacity
    self.motion_ids[index] = motion_ids
    self.frames[index] = frames
    self.z_nom[index] = z_nom
    self.sim_pos[index] = sim_pos
    self.sim_quat[index] = sim_quat
    self.sim_lin_vel[index] = sim_lin_vel
    self.sim_ang_vel[index] = sim_ang_vel
    self._cursor = int((self._cursor + n) % self.capacity)
    self.size = min(self.size + n, self.capacity)

  def sample(self, batch_size: int, device) -> dict[str, torch.Tensor]:
    n = min(batch_size, self.size)
    idx = torch.randperm(self.size, device=self.device)[:n]
    return {
      "motion_ids": self.motion_ids[idx],
      "frames": self.frames[idx],
      "sim_pos": self.sim_pos[idx],
      "sim_quat": self.sim_quat[idx],
      "sim_lin_vel": self.sim_lin_vel[idx],
      "sim_ang_vel": self.sim_ang_vel[idx],
      "z_nom": self.z_nom[idx],
    }

  def clear(self) -> None:
    self.size = 0
    self._cursor = 0


@dataclass(kw_only=True)
class RetargetCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  motion_files: tuple[str, ...] = ()
  source_bodies: tuple[str, ...] = ()
  target_bodies: tuple[str, ...] = ()
  quaternion_order: str = "wxyz"
  motion_weights: dict[str, float] = field(default_factory=dict)
  #: Builds (positions, quaternions_wxyz, standing_height) of the target bodies
  #: in the robot's nominal configuration.  A callable keeps this module free of
  #: a hard dependency on a specific robot.
  target_nominal: Callable[[ManagerBasedRlEnv], tuple[np.ndarray, np.ndarray, float]] | None = None
  #: Retargeting phase ramp; the reference is frozen while ``psi`` ramps up.
  #: Kept short (10 control steps) so the upper-level buffer is not starved by
  #: early terminations, which are frequent while the policy is still random.
  psi_ramp_s: float = 0.2
  joint_position_range: tuple[float, float] = (-0.1, 0.1)
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  adaptive_uniform_ratio: float = 0.1
  #: Upper-level knobs (Table 2 of the paper).
  tracking_reward: Literal["squared", "kernel"] = "squared"
  collect_upper: bool = True
  param_lr: float = 1.0e-4
  alpha: float = 0.0
  delta_pos: float = 0.5
  delta_ori: float = 0.5
  delta_z: float = 0.5
  loss_pos_weight: float = 10.0
  loss_ori_weight: float = 1.0
  loss_lin_vel_weight: float = 0.0
  loss_ang_vel_weight: float = 0.0
  upper_buffer_size: int = 16384
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> RetargetCommand:
    return RetargetCommand(self, env)