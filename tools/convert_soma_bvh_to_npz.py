"""Convert BONES-SEED SOMA BVH clips into the ReActor source-motion format.

Each output ``*.npz`` holds, for the sparse set of SOMA bodies that correspond to
robot bodies, the world-frame trajectory of that body at the policy rate:

===============  =========================================================
``fps``          control rate (50 Hz by default)
``body_names``   SOMA joint names, in correspondence order
``body_pos_w``   ``(T, B, 3)`` world positions in metres, Z-up
``body_quat_w``  ``(T, B, 4)`` world orientations, ``wxyz``
``body_lin_vel_w`` / ``body_ang_vel_w``  world velocities
``nominal_pos``  ``(B, 3)`` source body frames in the nominal configuration
``nominal_quat`` ``(B, 4)``
``z_nom``        per-clip vertical offset that puts the feet on the ground
``root_height``  source root height in the nominal configuration
===============  =========================================================

Calibration uses the fixed BONES-SEED uniform rig, independently of the input clips.

Usage::

  python tools/convert_soma_bvh_to_npz.py --src .tmp/soma_raw \\
      --dst src/assets/motions/reactor/soma
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from tools.soma_bvh import (  # noqa: E402
  BvhClip,
  angular_velocity,
  forward_kinematics,
  linear_velocity,
  parse_bvh,
  quat_from_matrix,
  resample,
  to_mujoco_frame,
)

# Semantic source -> target pairs live with the task config, but the converter
# only needs the SOMA side; keep this list in sync with ``CORRESPONDENCES``.
SOURCE_BODIES: tuple[str, ...] = (
  "Hips",
  "Chest",
  "Head",
  "LeftShoulder",
  "LeftArm",
  "LeftForeArm",
  "LeftHand",
  "RightShoulder",
  "RightArm",
  "RightForeArm",
  "RightHand",
  "LeftLeg",
  "LeftShin",
  "LeftFoot",
  "RightLeg",
  "RightShin",
  "RightFoot",
)

FOOT_BODIES: tuple[str, ...] = ("LeftFoot", "RightFoot")
ROOT_BODY = "Hips"
UNIFORM_RIG = _REPO_ROOT / "src/assets/motions/reactor/soma_base_skel_minimal.bvh"


@lru_cache(maxsize=1)
def uniform_skeleton() -> BvhClip:
  return parse_bvh(UNIFORM_RIG)


def normalize_proportions(clip: BvhClip) -> None:
  """Express a proportional SOMA performance on the training skeleton."""
  rig = uniform_skeleton()
  if [(j.name, j.parent) for j in clip.joints] != [(j.name, j.parent) for j in rig.joints]:
    raise ValueError("Expected the BONES-SEED SOMA BVH hierarchy")
  # In uniform files a few auxiliary offsets may vary slightly. The limb
  # lengths identify the published uniform proportions independently of pose.
  limbs = ("LeftShin", "RightShin", "LeftFoot", "RightFoot", "LeftHand", "RightHand")
  ids = [clip.joint_names.index(name) for name in limbs]
  lengths = np.array([np.linalg.norm(clip.joints[i].offset) for i in ids])
  uniform = np.array([np.linalg.norm(rig.joints[i].offset) for i in ids])
  if np.allclose(lengths, uniform, rtol=1e-4):
    return
  if np.any(lengths <= 0):
    raise ValueError("Degenerate SOMA limb length")
  stride_scale = uniform[:4].sum() / lengths[:4].sum()
  start = 0
  for joint, template in zip(clip.joints, rig.joints, strict=True):
    joint.offset = template.offset.copy()
    if joint.num_pos_channels:
      clip.motion[:, start:start + joint.num_pos_channels] *= stride_scale
    start += len(joint.channels)



def rig_calibration(rig: Path) -> dict:
  """Use the published SOMA bind-pose frame and the anatomical left axis."""
  clip = parse_bvh(rig)
  rot, pos = to_mujoco_frame(*forward_kinematics(clip))
  idx = [clip.joint_names.index(name) for name in SOURCE_BODIES]
  pos = pos[0, idx].copy()
  left = pos[SOURCE_BODIES.index("LeftLeg")] - pos[SOURCE_BODIES.index("RightLeg")]
  yaw = np.pi / 2 - np.arctan2(left[1], left[0])
  pos, quat = rotate_yaw(pos, quat_from_matrix(rot[0, idx]), yaw)
  pos[:, :2] -= pos[0, :2].copy()
  pos[:, 2] -= min(pos[SOURCE_BODIES.index(name), 2] for name in FOOT_BODIES)
  return dict(nominal_pos=pos, nominal_quat=quat, root_height=float(pos[0, 2]), yaw=float(yaw))


def rotate_yaw(
  pos: np.ndarray, quat: np.ndarray, yaw: float
) -> tuple[np.ndarray, np.ndarray]:
  """Rotate positions and ``wxyz`` quaternions about the world z axis."""
  if abs(yaw) < 1.0e-12:
    return pos, quat
  cos, sin = np.cos(yaw), np.sin(yaw)
  rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
  pos = pos @ rot.T
  half = 0.5 * yaw
  qy = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
  w1, x1, y1, z1 = qy
  w2, x2, y2, z2 = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
  out = np.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    axis=-1,
  )
  return pos, out


def convert_clip(
  path: Path,
  out_fps: float,
  nominal_pos: np.ndarray,
  nominal_quat: np.ndarray,
  root_height: float,
  yaw: float = 0.0,
) -> dict[str, np.ndarray]:
  names = list(SOURCE_BODIES)
  clip = parse_bvh(path)
  normalize_proportions(clip)
  if not np.isfinite(out_fps) or out_fps <= 0:
    raise ValueError("Output frame rate must be finite and positive")
  rot, pos = forward_kinematics(clip)
  rot, pos = to_mujoco_frame(rot, pos)

  index = [clip.joint_names.index(name) for name in names]
  pos = pos[:, index]
  quat = quat_from_matrix(rot[:, index])

  if abs(clip.fps - out_fps) > 1e-6:
    pos = resample(pos, clip.fps, out_fps)
    quat = resample(quat, clip.fps, out_fps, quaternion=True)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

  # Put the clip in the same world convention as the robot nominal (+X forward).
  pos, quat = rotate_yaw(pos, quat, yaw)

  pos[..., :2] -= pos[0, 0, :2].copy()
  lin_vel = linear_velocity(pos, out_fps)
  ang_vel = angular_velocity(quat, out_fps)

  # Per-clip vertical offset so the lowest foot sample sits on the ground
  # (the paper's ``z_nom``, following PHC).  The 2nd percentile is used instead
  # of the absolute minimum so a single noisy mocap frame cannot lift the whole
  # clip off the floor.
  foot_ids = [names.index(name) for name in FOOT_BODIES]
  z_nom = -float(np.percentile(pos[:, foot_ids, 2], 2.0))

  return {
    "fps": np.array([out_fps], dtype=np.float32),
    "body_names": np.array(names),
    "body_pos_w": pos.astype(np.float32),
    "body_quat_w": quat.astype(np.float32),
    "body_lin_vel_w": lin_vel.astype(np.float32),
    "body_ang_vel_w": ang_vel.astype(np.float32),
    "nominal_pos": nominal_pos.astype(np.float32),
    "nominal_quat": nominal_quat.astype(np.float32),
    "z_nom": np.array([z_nom], dtype=np.float32),
    "root_height": np.array([root_height], dtype=np.float32),
    "source_yaw": np.array(yaw),
    "quaternion_order": np.array("wxyz"),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--src", required=True, help="BVH file or directory.")
  parser.add_argument("--dst", required=True, help="Output directory for npz clips.")
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--limit", type=int, default=0, help="Convert at most N clips.")
  parser.add_argument("--rig", default=str(UNIFORM_RIG))
  args = parser.parse_args()

  src = Path(args.src).expanduser()
  paths = sorted(src.rglob("*.bvh")) if src.is_dir() else [src]
  if args.limit:
    paths = paths[: args.limit]
  if not paths:
    raise SystemExit(f"No BVH files under {src}")
  dst = Path(args.dst).expanduser()
  dst.mkdir(parents=True, exist_ok=True)

  calibration = rig_calibration(Path(args.rig))
  if len({p.stem for p in paths}) != len(paths):
    raise ValueError("Duplicate clip stems would overwrite an output")
  summary = []
  for path in paths:
    data = convert_clip(path, args.fps, **calibration)
    out = dst / f"{path.stem}.npz"
    np.savez(out, **data)
    frames = data["body_pos_w"].shape[0]
    summary.append(
      {
        "file": out.name,
        "source": str(path),
        "frames": int(frames),
        "duration_s": round(frames / args.fps, 2),
        "z_nom": round(float(data["z_nom"][0]), 4),
      }
    )
    print(f"  {out.name}: {frames} frames ({frames / args.fps:.2f} s)")

  (dst / "clips.json").write_text(
    json.dumps(
      {
        "fps": args.fps,
        "body_names": list(SOURCE_BODIES),
        "root_body": ROOT_BODY,
        "root_height": calibration["root_height"],
        "nominal_yaw_offset_deg": round(float(np.degrees(calibration["yaw"])), 3),
        "clips": summary,
      },
      indent=2,
    )
  )
  print(f"[INFO] Wrote {len(paths)} clips and clips.json to {dst}")


if __name__ == "__main__":
  main()
