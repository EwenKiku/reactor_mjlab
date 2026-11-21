"""Semantic SOMA -> Pi Plus body correspondences and the nominal target pose.

The paper only asks the user for a sparse set of rigid-body pairs.  For a
humanoid target every source body in :data:`SOURCE_BODIES` gets a counterpart;
the pairs are picked by structural similarity (hips->pelvis, chest->torso,
head->head, shoulder/arm/forearm/hand, thigh/shin/foot) with no iterative
refinement.

Calibration matches the published SOMA bind pose to a grounded Pi Plus T-pose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from collections.abc import Sequence

#: SOMA body name -> Pi Plus body name, in the order used by the retargeting
#: parameter tensors.  The first entry is the root pair.
BODY_PAIRS: tuple[tuple[str, str], ...] = (
  ("Hips", "base_link"),
  ("Chest", "torso_link"),
  ("Head", "head_pitch_link"),
  ("LeftShoulder", "l_shoulder_roll_link"),
  ("LeftArm", "l_upper_arm_link"),
  ("LeftForeArm", "l_elbow_link"),
  ("LeftHand", "l_wrist_link"),
  ("RightShoulder", "r_shoulder_roll_link"),
  ("RightArm", "r_upper_arm_link"),
  ("RightForeArm", "r_elbow_link"),
  ("RightHand", "r_wrist_link"),
  ("LeftLeg", "l_thigh_link"),
  ("LeftShin", "l_calf_link"),
  ("LeftFoot", "l_ankle_roll_link"),
  ("RightLeg", "r_thigh_link"),
  ("RightShin", "r_calf_link"),
  ("RightFoot", "r_ankle_roll_link"),
)

#: Bodies whose rotation only counts about "swing" because the robot has fewer
#: DoF than the human there (paper Sec. 5, swing/twist decomposition).  Pi Plus
#: tracks these with a 3-DoF ball joint, except the torso which has no yaw
#: actuator, so torso yaw is ignored.
IGNORE_TWIST_AXIS: dict[str, tuple[float, float, float]] = {
  "torso_link": (0.0, 0.0, 1.0),
}

SOURCE_BODIES: tuple[str, ...] = tuple(src for src, _ in BODY_PAIRS)
TARGET_BODIES: tuple[str, ...] = tuple(dst for _, dst in BODY_PAIRS)
ROOT_SOURCE = BODY_PAIRS[0][0]
ROOT_TARGET = BODY_PAIRS[0][1]
FEET_BODIES: tuple[str, str] = ("l_ankle_roll_link", "r_ankle_roll_link")

#: Match the arms-out SOMA bind pose. Unspecified hinges stay at MJCF qpos0.
NOMINAL_JOINT_POS = {"l_shoulder_roll_joint": np.pi / 2, "r_shoulder_roll_joint": -np.pi / 2}


def nominal_robot_frames(
  xml_path: str,
  body_names: Sequence[str],
  joint_pos: dict[str, float] | None = None,
  base_height: float = 0.351,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Nominal world frames of ``body_names`` for the robot, via MuJoCo FK.

  The root sits at ``(0, 0, base_height)`` on a flat floor.  Returns
  ``(positions, quaternions_wxyz, root_height_above_lowest_foot)``.
  """
  import re

  import mujoco
  from mjlab.utils.lab_api.math import quat_from_matrix

  import torch

  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)
  pos_map = NOMINAL_JOINT_POS if joint_pos is None else joint_pos
  for joint_id in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if name is None:
      continue
    for pattern, value in pos_map.items():
      if re.fullmatch(pattern, name):
        data.qpos[model.jnt_qposadr[joint_id]] = value
        break
  data.qpos[2] = base_height
  mujoco.mj_forward(model, data)

  positions = np.zeros((len(body_names), 3))
  quats = np.zeros((len(body_names), 4))
  for i, name in enumerate(body_names):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
      raise ValueError(f"{name!r} is not a body of {xml_path}")
    positions[i] = data.xpos[body_id]
    quats[i] = quat_from_matrix(
      torch.as_tensor(data.xmat[body_id].reshape(1, 3, 3), dtype=torch.float64)
    ).numpy()[0]

  foot_z = min(positions[list(body_names).index(name), 2] for name in FEET_BODIES)
  # Ground the reference using the actual cylindrical sole contact geometry.
  gids = [g for g in range(model.ngeom) if re.fullmatch(
    r"[lr]_(?:toe|heel)_.*", model.geom(g).name or "")]
  axis_z = np.abs(data.geom_xmat[gids, 8]).clip(0, 1)
  sole = (data.geom_xpos[gids, 2] - model.geom_size[gids, 1] * axis_z
          - model.geom_size[gids, 0] * np.sqrt(1 - axis_z**2)).min()
  positions[:, 2] -= sole
  root_height = float(base_height - foot_z)
  return positions, quats, root_height