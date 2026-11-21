"""Minimal SOMA BVH reader + forward kinematics.

BONES-SEED ships the motion capture data as BVH files on the SOMA skeleton
(``soma_base_skel_minimal.bvh`` hierarchy).  Every joint carries three
``Zrotation Yrotation Xrotation`` channels (the root adds ``Xposition
Yposition Zposition``); the frame rate is 120 Hz and the offsets are in
centimetres on a Y-up world.

This module is deliberately dependency-free (numpy only) so the retargeting
tools can run without MuJoCo:

* :func:`parse_bvh` returns the joint hierarchy and the raw channel matrix.
* :func:`forward_kinematics` turns that into per-frame global frames.
* :func:`to_mujoco_frame` re-expresses those frames in the Z-up, metre,
  +X-forward convention used by the robot models.

Channels are applied as ``R = Rz @ Ry @ Rx`` (BVH convention), so the local
joint rotation is an intrinsic Z-Y-X Euler sequence.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np

_JOINT_RE = re.compile(r"^(ROOT|JOINT)\s+(\S+)")
_OFFSET_RE = re.compile(r"^OFFSET\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)")
_CHANNELS_RE = re.compile(r"^CHANNELS\s+(\d+)\s+(.*)$")
_FRAMES_RE = re.compile(r"^Frames:\s+(\d+)")
_FRAME_TIME_RE = re.compile(r"^Frame Time:\s+([\d.eE+-]+)")

# Rotation from the BVH (Y-up) world to the robot convention (Z-up, +X forward,
# +Y left).  SOMA's world has +Y up, -X to the character's left and -Z forward,
# so ``mj = (-z, -x, y)`` maps left/forward/up onto +Y/+X/+Z respectively.
_BVH_TO_ZUP = np.array(
  [
    [0.0, 0.0, -1.0],  # mj x (forward) = -bvh z
    [-1.0, 0.0, 0.0],  # mj y (left)    = -bvh x
    [0.0, 1.0, 0.0],  # mj z (up)      =  bvh y
  ]
)


@dataclasses.dataclass
class Joint:
  name: str
  parent: int | None
  offset: np.ndarray  # (3,) local translation in the parent frame, BVH units
  channels: tuple[str, ...]

  @property
  def num_rot_channels(self) -> int:
    return sum(1 for c in self.channels if c.lower().endswith("rotation"))

  @property
  def num_pos_channels(self) -> int:
    return sum(1 for c in self.channels if c.lower().endswith("position"))


@dataclasses.dataclass
class BvhClip:
  joints: list[Joint]
  frame_time: float
  motion: np.ndarray  # (frames, sum(num_channels))

  @property
  def fps(self) -> float:
    return 1.0 / self.frame_time

  @property
  def joint_names(self) -> list[str]:
    return [j.name for j in self.joints]

  def channel_slice(self, joint: Joint) -> tuple[int, int]:
    start = sum(len(j.channels) for j in self.joints[: self.joints.index(joint)])
    return start, start + len(joint.channels)


def _parse_hierarchy(tokens: list[str]) -> list[Joint]:
  """Recursive-descent parse of the HIERARCHY block.

  ``End Site`` blocks carry an ignored ``OFFSET`` and must not be confused with
  their parent joint, so they are parsed and dropped explicitly.
  """
  joints: list[Joint] = []
  pos = 0

  def parse_node(parent: int | None) -> None:
    nonlocal pos
    kind = tokens[pos]
    pos += 1
    if kind == "End":
      pos += 1  # "Site"
      if pos < len(tokens) and tokens[pos] == "{":
        pos += 1
        while pos < len(tokens) and tokens[pos] != "}":
          pos += 1
        pos += 1
      return
    name = tokens[pos]
    pos += 1
    joint = Joint(name=name, parent=parent, offset=np.zeros(3), channels=())
    index = len(joints)
    joints.append(joint)
    if pos < len(tokens) and tokens[pos] == "{":
      pos += 1
      while pos < len(tokens):
        token = tokens[pos]
        if token == "}":
          pos += 1
          break
        if token == "OFFSET":
          joint.offset = np.array([float(v) for v in tokens[pos + 1 : pos + 4]])
          pos += 4
        elif token == "CHANNELS":
          count = int(tokens[pos + 1])
          joint.channels = tuple(tokens[pos + 2 : pos + 2 + count])
          pos += 2 + count
        elif token in ("ROOT", "JOINT", "End"):
          parse_node(index)
        else:
          raise ValueError(f"Unexpected token {token!r} in HIERARCHY")
    return

  while pos < len(tokens):
    if tokens[pos] == "HIERARCHY":
      pos += 1
      continue
    parse_node(None)
  return joints


def parse_bvh(path: str | Path) -> BvhClip:
  """Parse a BVH file into joints plus the raw channel matrix."""
  text = Path(path).read_text(errors="replace")
  lines = text.splitlines()
  motion_start = next(
    (i for i, line in enumerate(lines) if line.strip() == "MOTION"), None
  )
  if motion_start is None:
    raise ValueError(f"{path}: no MOTION section")
  hierarchy = " ".join(lines[:motion_start]).replace("{", " { ").replace("}", " } ")
  joints = _parse_hierarchy(hierarchy.split())

  frame_time = 0.0
  num_frames = 0
  motion_rows: list[list[float]] = []
  for line in lines[motion_start + 1 :]:
    stripped = line.strip()
    if not stripped:
      continue
    match = _FRAMES_RE.match(stripped)
    if match:
      num_frames = int(match.group(1))
      continue
    match = _FRAME_TIME_RE.match(stripped)
    if match:
      frame_time = float(match.group(1))
      continue
    motion_rows.append([float(v) for v in stripped.split()])

  if not joints:
    raise ValueError(f"No joints parsed from {path}")
  if num_frames and len(motion_rows) != num_frames:
    raise ValueError(
      f"{path}: header says {num_frames} frames, parsed {len(motion_rows)}"
    )
  motion = np.array(motion_rows, dtype=np.float64)
  if not np.isfinite(frame_time) or frame_time <= 0 or motion.ndim != 2 or not np.isfinite(motion).all():
    raise ValueError(f"{path}: invalid frame time or motion samples")
  return BvhClip(joints=joints, frame_time=frame_time, motion=motion)


def _euler_zyx(angles_deg: np.ndarray) -> np.ndarray:
  """`Zrotation Yrotation Xrotation` -> rotation matrices ``Rz @ Ry @ Rx``."""
  z, y, x = np.deg2rad(angles_deg).transpose(1, 0)
  cz, sz = np.cos(z), np.sin(z)
  cy, sy = np.cos(y), np.sin(y)
  cx, sx = np.cos(x), np.sin(x)
  out = np.empty((angles_deg.shape[0], 3, 3))
  out[:, 0, 0] = cz * cy
  out[:, 0, 1] = cz * sy * sx - sz * cx
  out[:, 0, 2] = cz * sy * cx + sz * sx
  out[:, 1, 0] = sz * cy
  out[:, 1, 1] = sz * sy * sx + cz * cx
  out[:, 1, 2] = sz * sy * cx - cz * sx
  out[:, 2, 0] = -sy
  out[:, 2, 1] = cy * sx
  out[:, 2, 2] = cy * cx
  return out


def local_transforms(clip: BvhClip) -> tuple[np.ndarray, np.ndarray]:
  """Per-frame local rotation and translation of every joint."""
  num_frames = clip.motion.shape[0]
  num_joints = len(clip.joints)
  rot = np.zeros((num_frames, num_joints, 3, 3))
  trans = np.zeros((num_frames, num_joints, 3))
  offset = 0
  for j, joint in enumerate(clip.joints):
    rotation = ("Zrotation", "Yrotation", "Xrotation")
    if joint.channels not in (rotation, ("Xposition", "Yposition", "Zposition") + rotation):
      raise ValueError(f"{joint.name}: unsupported SOMA BVH channels {joint.channels}")
    count = len(joint.channels)
    block = clip.motion[:, offset : offset + count]
    offset += count
    trans[:, j] = joint.offset
    rot[:, j] = np.broadcast_to(np.eye(3), (num_frames, 3, 3))
    if joint.num_pos_channels:
      # BONES-SEED writes translation channels that already *contain* the rest
      # offset (Hips position == Hips OFFSET in the standing frame), so they
      # replace the offset rather than being added on top of it.
      trans[:, j] = block[:, : joint.num_pos_channels]
    if joint.num_rot_channels:
      rot[:, j] = _euler_zyx(block[:, joint.num_pos_channels :])
  if offset != clip.motion.shape[1]:
    raise ValueError(
      f"channel count mismatch: parsed {offset}, motion has {clip.motion.shape[1]}"
    )
  return rot, trans


def forward_kinematics(clip: BvhClip) -> tuple[np.ndarray, np.ndarray]:
  """Global rotation ``(frames, joints, 3, 3)`` and position ``(frames, joints, 3)``.

  Positions are in BVH world units (centimetres), Y-up.
  """
  local_rot, local_trans = local_transforms(clip)
  num_frames, num_joints = local_rot.shape[:2]
  rot = np.zeros_like(local_rot)
  pos = np.zeros((num_frames, num_joints, 3))
  for j, joint in enumerate(clip.joints):
    if joint.parent is None:
      rot[:, j] = local_rot[:, j]
      pos[:, j] = local_trans[:, j]
    else:
      p = joint.parent
      rot[:, j] = rot[:, p] @ local_rot[:, j]
      pos[:, j] = pos[:, p] + (rot[:, p] @ local_trans[:, j][..., None])[..., 0]
  return rot, pos


def to_mujoco_frame(rot: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """BVH (Y-up, cm) global frames -> MuJoCo (Z-up, m) global frames.

  A change of basis for rotations is a similarity transform ``M R M^T``.
  """
  rot_out = _BVH_TO_ZUP[None, None] @ rot @ _BVH_TO_ZUP.T[None, None]
  pos_out = (pos @ _BVH_TO_ZUP.T) * 0.01
  return rot_out, pos_out


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
  """Rotation matrices ``(..., 3, 3)`` -> ``wxyz`` quaternions ``(..., 4)``."""
  m = np.asarray(matrix)
  m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
  m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
  m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
  trace = m00 + m11 + m22
  quat = np.empty(m.shape[:-2] + (4,), dtype=np.float64)
  # Case 1: trace is largest.
  cond0 = trace > 0.0
  s = np.sqrt(np.where(cond0, trace + 1.0, 1.0)) * 2.0
  quat[..., 0] = np.where(cond0, 0.25 * s, 0.0)
  quat[..., 1] = np.where(cond0, (m21 - m12) / s, 0.0)
  quat[..., 2] = np.where(cond0, (m02 - m20) / s, 0.0)
  quat[..., 3] = np.where(cond0, (m10 - m01) / s, 0.0)
  # Case 2: m00 is largest.
  cond1 = (~cond0) & (m00 > m11) & (m00 > m22)
  s = np.sqrt(np.where(cond1, 1.0 + m00 - m11 - m22, 1.0)) * 2.0
  quat[..., 0] = np.where(cond1, (m21 - m12) / s, quat[..., 0])
  quat[..., 1] = np.where(cond1, 0.25 * s, quat[..., 1])
  quat[..., 2] = np.where(cond1, (m01 + m10) / s, quat[..., 2])
  quat[..., 3] = np.where(cond1, (m02 + m20) / s, quat[..., 3])
  # Case 3: m11 is largest.
  cond2 = (~cond0) & (~cond1) & (m11 > m22)
  s = np.sqrt(np.where(cond2, 1.0 + m11 - m00 - m22, 1.0)) * 2.0
  quat[..., 0] = np.where(cond2, (m02 - m20) / s, quat[..., 0])
  quat[..., 1] = np.where(cond2, (m01 + m10) / s, quat[..., 1])
  quat[..., 2] = np.where(cond2, 0.25 * s, quat[..., 2])
  quat[..., 3] = np.where(cond2, (m12 + m21) / s, quat[..., 3])
  # Case 4: m22 is largest.
  cond3 = (~cond0) & (~cond1) & (~cond2)
  s = np.sqrt(np.where(cond3, 1.0 + m22 - m00 - m11, 1.0)) * 2.0
  quat[..., 0] = np.where(cond3, (m10 - m01) / s, quat[..., 0])
  quat[..., 1] = np.where(cond3, (m02 + m20) / s, quat[..., 1])
  quat[..., 2] = np.where(cond3, (m12 + m21) / s, quat[..., 2])
  quat[..., 3] = np.where(cond3, 0.25 * s, quat[..., 3])
  # Normalise and keep the scalar part positive for a stable interpolation.
  quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
  quat *= np.where(quat[..., :1] < 0.0, -1.0, 1.0)
  return quat


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
  """Spherical interpolation between ``wxyz`` quaternions.

  ``q0`` and ``q1`` are ``(F, ...)`` and ``t`` is ``(F_out, 1)``; ``t`` is
  broadcast against the trailing (non-frame) axes and the result takes the
  output frame count on axis 0.
  """
  q0, q1 = np.broadcast_arrays(q0, q1)
  t = np.asarray(t, dtype=np.float64)
  if t.ndim == 1:
    t = t[:, None]
  # Broadcast t only over the frame axis.
  t = t.reshape((t.shape[0],) + (1,) * (q0.ndim - 1))
  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1 = np.where(dot < 0.0, -q1, q1)
  dot = np.abs(dot)
  linear = dot > 0.9995
  theta = np.arccos(np.clip(dot, -1.0, 1.0))
  sin_theta = np.sin(theta)
  w0 = np.where(
    linear, 1.0 - t, np.sin((1.0 - t) * theta) / np.maximum(sin_theta, 1e-9)
  )
  w1 = np.where(linear, t, np.sin(t * theta) / np.maximum(sin_theta, 1e-9))
  out = w0 * q0 + w1 * q1
  return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample(
  values: np.ndarray,
  src_fps: float,
  dst_fps: float,
  quaternion: bool = False,
) -> np.ndarray:
  """Linear (or slerp, when ``quaternion``) resampling along axis 0."""
  num_frames = values.shape[0]
  if abs(src_fps - dst_fps) < 1e-9 or num_frames < 2:
    return values
  duration = (num_frames - 1) / src_fps
  num_out = int(round(duration * dst_fps)) + 1
  grid = np.minimum(np.arange(num_out) / dst_fps * src_fps, num_frames - 1)
  index = np.floor(grid).astype(np.int64)
  index = np.clip(index, 0, num_frames - 2)
  frac = (grid - index)[:, None]
  a, b = values[index], values[index + 1]
  if quaternion:
    return quat_slerp(a, b, frac[:, 0])
  while frac.ndim < a.ndim:
    frac = frac[..., None]
  return a + (b - a) * frac


def angular_velocity(
  quat: np.ndarray, fps: float, wrap: bool = True
) -> np.ndarray:
  """World-frame angular velocity ``(frames, 3)`` from ``wxyz`` quaternions."""
  num_frames = quat.shape[0]
  if num_frames < 2:
    return np.zeros(quat.shape[:-1] + (3,))
  inv = np.zeros_like(quat)
  inv[..., 0] = quat[..., 0]
  inv[..., 1:] = -quat[..., 1:]
  delta = _quat_mul(quat[1:], inv[:-1])
  delta *= np.where(delta[..., :1] < 0.0, -1.0, 1.0)
  angle = 2.0 * np.arccos(np.clip(delta[..., 0], -1.0, 1.0))
  sin_half = np.sqrt(np.maximum(1.0 - delta[..., 0] ** 2, 0.0))
  axis = delta[..., 1:] / np.maximum(sin_half[..., None], 1e-9)
  omega = axis * (angle / (1.0) * fps)[..., None]
  if wrap:
    omega = omega * (angle > 1e-6)[..., None]
  # Forward difference: repeat the last sample.
  return np.concatenate([omega, omega[-1:]], axis=0)


def linear_velocity(pos: np.ndarray, fps: float) -> np.ndarray:
  """World-frame linear velocity ``(frames, 3)`` by forward difference."""
  num_frames = pos.shape[0]
  if num_frames < 2:
    return np.zeros_like(pos)
  vel = (pos[1:] - pos[:-1]) * fps
  return np.concatenate([vel, vel[-1:]], axis=0)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
  return np.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    axis=-1,
  )
