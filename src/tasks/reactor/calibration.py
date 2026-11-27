"""Per-robot link map and NPZ-to-link rotations loaded from ``calibration.yaml``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
  from collections.abc import Sequence


class RobotCalibration:
  """Link pairs and the nominal pose used for position calibration.

  ``npz_to_robot[i]`` maps a vector in the i-th NPZ body frame into that
  pair's robot link frame: ``v_robot = R @ v_npz``.
  """

  def __init__(
    self,
    source_bodies: tuple[str, ...],
    target_bodies: tuple[str, ...],
    npz_to_robot: np.ndarray,
    nominal_joints: dict[str, float],
    base_height: float,
    foot_bodies: tuple[str, ...],
    foot_geoms: str,
  ) -> None:
    self.source_bodies = source_bodies
    self.target_bodies = target_bodies
    self.npz_to_robot = npz_to_robot
    self.nominal_joints = nominal_joints
    self.base_height = base_height
    self.foot_bodies = foot_bodies
    self.foot_geoms = foot_geoms

  @property
  def root_body(self) -> str:
    return self.target_bodies[0]


def _rotation(raw: object, label: str) -> np.ndarray:
  matrix = np.asarray(raw, dtype=np.float64)
  if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
    raise ValueError(f"{label} rotation must be a finite 3x3 matrix")
  if abs(float(np.linalg.det(matrix)) - 1.0) > 1e-3:
    raise ValueError(f"{label} rotation determinant is not 1")
  error = np.max(np.abs(matrix.T @ matrix - np.eye(3)))
  if error > 1e-3:
    raise ValueError(f"{label} rotation is not orthonormal")
  return matrix


def load_calibration(path: Path) -> RobotCalibration:
  """Read ``calibration.yaml`` next to a robot's ``env_cfg.py``."""
  with Path(path).open() as handle:
    data = yaml.safe_load(handle)
  if not isinstance(data, dict) or not data.get("links"):
    raise ValueError(f"{path} needs a non-empty links list")
  sources: list[str] = []
  targets: list[str] = []
  rotations: list[np.ndarray] = []
  for entry in data["links"]:
    source = str(entry["source"])
    target = str(entry["target"])
    sources.append(source)
    targets.append(target)
    rotations.append(_rotation(entry["rotation"], f"{source}->{target}"))
  if sources[0] != "Hips":
    raise ValueError(f"{path} must list Hips first; the root pair is index 0")
  if len(set(sources)) != len(sources):
    raise ValueError(f"{path} repeats a source body")
  feet = tuple(str(name) for name in data["foot_bodies"])
  missing = sorted(set(feet) - set(targets))
  if not feet or missing:
    raise ValueError(f"{path} foot_bodies must be listed targets, missing {missing}")
  joints = {str(name): float(value) for name, value in (data.get("nominal_joints") or {}).items()}
  return RobotCalibration(
    source_bodies=tuple(sources),
    target_bodies=tuple(targets),
    npz_to_robot=np.stack(rotations),
    nominal_joints=joints,
    base_height=float(data["base_height"]),
    foot_bodies=feet,
    foot_geoms=str(data["foot_geoms"]),
  )


def nominal_positions(
  xml_path: str,
  body_names: Sequence[str],
  joint_pos: dict[str, float],
  base_height: float,
  foot_bodies: Sequence[str],
  foot_geom_expr: str,
) -> tuple[np.ndarray, float]:
  """World positions of ``body_names`` with the feet placed on the ground.

  Returns ``(positions, root_height)``. ``root_height`` is the configured base
  height above the lowest foot body, before the sole shift.
  """
  import mujoco

  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)
  for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
      continue
    for pattern, value in joint_pos.items():
      if re.fullmatch(pattern, name):
        data.qpos[model.jnt_qposadr[joint_id]] = value
        break
  data.qpos[2] = base_height
  mujoco.mj_forward(model, data)

  positions = np.zeros((len(body_names), 3))
  names = list(body_names)
  for i, name in enumerate(names):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
      raise ValueError(f"{name!r} is not a body of {xml_path}")
    positions[i] = data.xpos[body_id]

  foot_z = min(positions[names.index(name), 2] for name in foot_bodies)
  gids = [
    g for g in range(model.ngeom) if re.fullmatch(foot_geom_expr, model.geom(g).name or "")
  ]
  if not gids:
    raise ValueError(f"No foot geoms match {foot_geom_expr!r} in {xml_path}")
  axis_z = np.abs(data.geom_xmat[gids, 8]).clip(0, 1)
  sole = (
    data.geom_xpos[gids, 2]
    - model.geom_size[gids, 1] * axis_z
    - model.geom_size[gids, 0] * np.sqrt(np.maximum(1.0 - axis_z**2, 0.0))
  ).min()
  positions[:, 2] -= sole
  return positions, float(base_height - foot_z)
