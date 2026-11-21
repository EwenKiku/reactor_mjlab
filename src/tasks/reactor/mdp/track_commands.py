"""Multi-clip motion command for the downstream tracking task.

``mjlab.tasks.tracking.MotionCommand`` loads exactly one ``.npz``.  The ReActor
pipeline emits one clip per source motion, so this term keeps a clip library:
every environment tracks its own clip and advances its own frame index.

The interface mirrors the mjlab tracking command (``anchor_*``,
``body_*_relative_w``, ``robot_*``) so the stock tracking rewards, observations
and terminations keep working.  Sampling draws a clip first and then a frame
inside that clip's own failure histogram, which is what the retargeting command
also does.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_QUAT_TO_WXYZ = {"wxyz": (0, 1, 2, 3), "xyzw": (3, 0, 1, 2)}


class ClipLibrary:
  """Every clip under the given paths, concatenated along time."""

  def __init__(
    self,
    motion_files: tuple[str, ...],
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    device: str | torch.device = "cpu",
    motion_weights: dict[str, float] | None = None,
  ) -> None:
    paths = self._expand(motion_files)
    clips = [self._load(path, joint_names, body_names, device) for path in paths]
    self.names = tuple(Path(p).stem for p in paths)
    self.num_motions = len(clips)
    self.lengths = torch.tensor(
      [c["joint_pos"].shape[0] for c in clips], dtype=torch.long, device=device
    )
    self.starts = torch.zeros(self.num_motions, dtype=torch.long, device=device)
    if self.num_motions > 1:
      self.starts[1:] = torch.cumsum(self.lengths[:-1], dim=0)
    for key in (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      setattr(self, key, torch.cat([c[key] for c in clips]))
    self.fps = float(clips[0]["fps"])
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
  def _expand(motion_files: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for raw in motion_files:
      path = Path(raw).expanduser()
      if path.is_dir():
        found = sorted(path.glob("*.npz"))
        if not found:
          raise FileNotFoundError(f"No .npz clips in {path}")
        paths.extend(str(item) for item in found)
      else:
        paths.append(str(path))
    if not paths:
      raise ValueError("No motion files configured.")
    return paths

  @staticmethod
  def _load(
    path: str,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
    device: str | torch.device,
  ) -> dict[str, torch.Tensor | float]:
    """Load one clip and remap it onto ``joint_names`` / ``body_names``.

    Clip files only carry the actuated joints (the head is driven by a spring in
    the robot config), so joints missing from the file are zero-filled.  Bodies
    must all be present: a missing body would silently change the tracking
    reward's normalisation.
    """
    data = np.load(path)
    order = "wxyz"
    if "quaternion_order" in data.files:
      order = str(np.asarray(data["quaternion_order"]).item())
    if order not in _QUAT_TO_WXYZ:
      raise ValueError(f"Unsupported quaternion order {order!r} in {path}")

    src_joints = [str(n) for n in data["joint_names"]]
    src_bodies = [str(n) for n in data["body_names"]]
    missing_b = sorted(set(body_names) - set(src_bodies))
    if missing_b:
      raise ValueError(f"{path} is missing bodies {missing_b}")
    bidx = [src_bodies.index(name) for name in body_names]

    def load(key: str) -> torch.Tensor:
      return torch.as_tensor(data[key], dtype=torch.float32, device=device)

    def load_joints(key: str) -> torch.Tensor:
      raw = load(key)
      out = torch.zeros(raw.shape[0], len(joint_names), device=device)
      present = [i for i, name in enumerate(joint_names) if name in src_joints]
      out[:, present] = raw[:, [src_joints.index(joint_names[i]) for i in present]]
      return out

    return {
      "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
      "joint_pos": load_joints("joint_pos"),
      "joint_vel": load_joints("joint_vel"),
      "body_pos_w": load("body_pos_w")[:, bidx],
      "body_quat_w": load("body_quat_w")[:, bidx][..., list(_QUAT_TO_WXYZ[order])],
      "body_lin_vel_w": load("body_lin_vel_w")[:, bidx],
      "body_ang_vel_w": load("body_ang_vel_w")[:, bidx],
    }

  def frame_index(self, motion_ids: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    return self.starts[motion_ids] + frames

  def sample_motion(self, count: int, device: str | torch.device) -> torch.Tensor:
    return torch.multinomial(self.sample_prob, count, replacement=True).to(device)


class TrackingCommand(CommandTerm):
  """Track one retargeted clip per environment."""

  cfg: TrackingCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: TrackingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.motion = ClipLibrary(
      cfg.motion_files,
      tuple(self.robot.joint_names),
      tuple(cfg.body_names),
      device=self.device,
      motion_weights=cfg.motion_weights,
    )
    control_fps = 1.0 / env.step_dt
    if abs(self.motion.fps - control_fps) > 1e-3:
      raise ValueError(
        f"Clips are {self.motion.fps} fps but the policy runs at {control_fps} Hz."
      )

    self.motion_ids = self.motion.sample_motion(self.num_envs, self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0
    self._pending_forward = False

    # One failure bin per control second, per clip.
    self.bin_counts = self.motion.lengths // int(control_fps) + 1
    self.bin_failed_count = torch.zeros(
      self.motion.num_motions, int(self.bin_counts.max()), device=self.device
    )
    self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
    kernel = torch.tensor(
      [cfg.adaptive_lambda**i for i in range(cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = kernel / kernel.sum()

    for name in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_body_lin_vel",
      "error_body_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
      "sampling_entropy",
      "sampling_top1_prob",
      "sampling_top1_bin",
    ):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
    print(
      "[INFO] Tracking clips: "
      + ", ".join(
        f"{name} ({int(length)} frames)"
        for name, length in zip(
          self.motion.names, self.motion.lengths.tolist(), strict=True
        )
      )
    )

  ##
  # Reference views.
  ##

  @property
  def _row(self) -> torch.Tensor:
    return self.motion.frame_index(self.motion_ids, self.time_steps)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self._row]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self._row]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self.motion.body_pos_w[self._row] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self._row]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self._row]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self._row]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return self.body_pos_w[:, self.motion_anchor_body_index]

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.body_quat_w[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.body_lin_vel_w[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.body_ang_vel_w[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

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
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  ##
  # Metrics.
  ##

  def _update_metrics(self) -> None:
    from mjlab.utils.lab_api.math import quat_error_magnitude

    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  ##
  # Sampling.
  ##

  def _clip_lengths(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    ids = self.motion_ids if env_ids is None else self.motion_ids[env_ids]
    return self.motion.lengths[ids]

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    terminated = self._env.termination_manager.terminated[env_ids]
    failed = env_ids[terminated]
    if failed.numel():
      mids = self.motion_ids[failed]
      lengths = self._clip_lengths(failed).clamp(min=1)
      counts = self.bin_counts[mids]
      bins = torch.clamp(self.time_steps[failed] * counts // lengths, max=counts - 1)
      self._current_bin_failed.zero_()
      for mid in mids.unique().tolist():
        count = int(self.bin_counts[mid])
        selected = bins[mids == mid]
        self._current_bin_failed[mid, :count] = torch.bincount(
          selected, minlength=count
        ).float()
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1.0 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )

    self.motion_ids[env_ids] = self.motion.sample_motion(len(env_ids), self.device)
    for mid in self.motion_ids[env_ids].unique().tolist():
      selected = env_ids[self.motion_ids[env_ids] == mid]
      count = int(self.bin_counts[mid])
      length = int(self.motion.lengths[mid])
      probs = (
        self.bin_failed_count[mid, :count]
        + self.cfg.adaptive_uniform_ratio / float(count)
      )
      probs = torch.nn.functional.pad(
        probs.view(1, 1, -1), (0, self.cfg.adaptive_kernel_size - 1), mode="replicate"
      )
      probs = torch.nn.functional.conv1d(probs, self.kernel.view(1, 1, -1)).view(-1)
      probs = probs / probs.sum()
      sampled = torch.multinomial(probs, selected.numel(), replacement=True)
      self.time_steps[selected] = (
        (sampled + sample_uniform(0.0, 1.0, (selected.numel(),), device=self.device))
        / count
        * (length - 1)
      ).long()
      entropy = -(probs * (probs + 1e-12).log()).sum()
      self.metrics["sampling_entropy"][selected] = (
        entropy / math.log(count) if count > 1 else 1.0
      )
      top_prob, top_bin = probs.max(dim=0)
      self.metrics["sampling_top1_prob"][selected] = top_prob
      self.metrics["sampling_top1_bin"][selected] = top_bin.float() / count

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    self.motion_ids[env_ids] = self.motion.sample_motion(len(env_ids), self.device)
    lengths = self._clip_lengths(env_ids).clamp(min=1)
    self.time_steps[env_ids] = (
      torch.rand(len(env_ids), device=self.device) * lengths.float()
    ).long().clamp(max=lengths - 1)
    self.metrics["sampling_entropy"][env_ids] = 1.0
    self.metrics["sampling_top1_prob"][env_ids] = 0.5
    self.metrics["sampling_top1_bin"][env_ids] = 0.5

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    soft = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft[:, :, 0], soft[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return
    if self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    else:
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    def sample_range(ranges: dict, size: int) -> torch.Tensor:
      table = torch.tensor(
        [ranges.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=self.device,
      )
      return sample_uniform(table[:, 0], table[:, 1], (size, 6), device=self.device)

    pose = sample_range(self.cfg.pose_range, len(env_ids))
    root_pos = root_pos + pose[:, :3]
    root_ori = quat_mul(
      quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]), root_ori
    )
    vel = sample_range(self.cfg.velocity_range, len(env_ids))
    root_lin_vel = root_lin_vel + vel[:, :3]
    root_ang_vel = root_ang_vel + vel[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]
    joint_pos = joint_pos + sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      joint_pos.shape,
      device=self.device,
    )
    self._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )
    self._pending_forward = True

  def update_relative_body_poses(self) -> None:
    """Body poses expressed in the robot anchor frame (yaw-aligned, height kept)."""
    num_bodies = len(self.cfg.body_names)
    anchor_pos = self.anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
    anchor_quat = self.anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
    robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
    robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)

    delta_pos = robot_anchor_pos.clone()
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))

    self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
    self.body_pos_relative_w = delta_pos + quat_apply(
      delta_ori, self.body_pos_w - anchor_pos
    )

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      self.time_steps += 1
    else:
      self.time_steps[env_ids] += 1
    ended = self.time_steps >= self._clip_lengths()
    if torch.any(ended):
      self._resample_command(ended.nonzero().flatten())
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    lengths = self._clip_lengths(env_ids)
    self.time_steps[env_ids] = torch.clamp(
      torch.full_like(lengths, frame), max=lengths - 1
    )
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )
    self._pending_forward = True


@dataclasses.dataclass(kw_only=True)
class TrackingCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  motion_files: tuple[str, ...] = ()
  """Files or directories; a directory contributes every ``*.npz`` in it."""
  motion_weights: dict[str, float] = dataclasses.field(default_factory=dict)
  anchor_body_name: str = ""
  body_names: tuple[str, ...] = ()
  pose_range: dict[str, tuple[float, float]] = dataclasses.field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = dataclasses.field(
    default_factory=dict
  )
  joint_position_range: tuple[float, float] = (-0.1, 0.1)
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> TrackingCommand:
    return TrackingCommand(self, env)