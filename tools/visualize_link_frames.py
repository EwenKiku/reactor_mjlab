"""Draw every BVH joint frame next to the converted NPZ body frames.

The left figure is the raw BVH pose. Its red/green/blue axes are the BVH joint
axes (local X/Y/Z). The skeleton is only turned upright so MuJoCo's Z-up world
can show it; those axes are not the training basis.

The right figure is the NPZ written by the training converter. Its axes are
``body_quat_w`` / ``nominal_quat``: Z-up, +X forward, local axes after the
BVH-to-MuJoCo change of basis. That is the frame used when a robot is calibrated.

Usage::

  python -m tools.visualize_link_frames
  python -m tools.visualize_link_frames --bvh /path/to/motion.bvh --out outputs/link_frames/motion.npz
"""

from __future__ import annotations

import argparse
import queue
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
  from .convert_soma_bvh_to_npz import SOURCE_BODIES, UNIFORM_RIG, convert_clip, rig_calibration
  from .soma_bvh import _BVH_TO_ZUP, forward_kinematics, parse_bvh
else:
  from convert_soma_bvh_to_npz import SOURCE_BODIES, UNIFORM_RIG, convert_clip, rig_calibration
  from soma_bvh import _BVH_TO_ZUP, forward_kinematics, parse_bvh

REPO = Path(__file__).resolve().parents[1]
AXIS_COLORS = ([1.0, 0.15, 0.12, 1.0], [0.15, 0.85, 0.2, 1.0], [0.25, 0.4, 1.0, 1.0])


def _yaw_matrix(yaw: float) -> np.ndarray:
  cos, sin = np.cos(yaw), np.sin(yaw)
  return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def bvh_view(pos_cm: np.ndarray, rot: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Upright positions, original BVH axes, and converted training axes.

  ``pos_cm`` is ``(F, J, 3)`` centimetres, Y-up. Both rotations are ``(F, J, 3, 3)``
  with local axes in the columns, expressed in the standing Z-up world.
  """
  basis = _BVH_TO_ZUP
  upright = _yaw_matrix(yaw) @ basis
  pos = (pos_cm @ basis.T) * 0.01
  pos = pos @ _yaw_matrix(yaw).T
  # Columns stay the local axes. Original keeps BVH X/Y/Z; converted applies M R M^T.
  original = np.einsum("ip,fnpq->fniq", upright, rot)
  converted = np.einsum("ip,fnpq,qk->fnik", upright, rot, basis.T)
  return pos, original, converted


def edges_of(names: list[str], parents: list[int | None]) -> list[tuple[int, int]]:
  pairs = []
  for joint, parent in enumerate(parents):
    if parent is not None and names[parent] != "Root":
      pairs.append((parent, joint))
  return pairs


def prepare(bvh: Path, out: Path, rig: Path, fps: float) -> dict:
  calibration = rig_calibration(rig)
  clip = parse_bvh(bvh)
  rot, pos = forward_kinematics(clip)
  view_pos, original, converted = bvh_view(pos, rot, calibration["yaw"])
  hip = clip.joint_names.index("Hips")
  feet = [clip.joint_names.index(name) for name in ("LeftFoot", "RightFoot")]
  view_pos = view_pos.copy()
  view_pos[..., :2] -= view_pos[0, hip, :2]
  # Same constant the converter stores in the nominal pose: feet of frame 0 on the floor.
  view_pos[..., 2] -= np.min(view_pos[0, feet, 2])
  payload = convert_clip(bvh, fps, **calibration)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(out, **payload)
  names = clip.joint_names
  source_ids = [names.index(name) for name in SOURCE_BODIES]
  npz_rot = Rotation.from_quat(payload["body_quat_w"][0], scalar_first=True).as_matrix()
  # Frame 0 is not resampled away. Proportional clips change bone lengths, not channels.
  error = np.linalg.norm(converted[0, source_ids] - npz_rot, axis=(1, 2)).max()
  print(f"已写入 {out}")
  print(f"BVH {bvh.name}: {len(names)} 个关节，{len(clip.motion)} 帧 @ {clip.fps:g} Hz")
  print(f"NPZ: {payload['body_pos_w'].shape[0]} 帧 @ {fps:g} Hz，17 个刚体")
  print(f"第 0 帧转换轴与 NPZ body_quat_w 的最大矩阵差 {error:.3g}")
  parents = [joint.parent for joint in clip.joints]
  return dict(
    names=names,
    source=list(SOURCE_BODIES),
    source_ids=source_ids,
    parents=parents,
    edges=edges_of(names, parents),
    bvh_pos=view_pos,
    original=original,
    converted=converted,
    bvh_fps=clip.fps,
    pos=payload["body_pos_w"],
    quat=payload["body_quat_w"],
    nominal_pos=payload["nominal_pos"],
    nominal_quat=payload["nominal_quat"],
    z_nom=float(payload["z_nom"][0]),
    fps=fps,
    npz=out.name,
    bvh=bvh.name,
  )


def _connector(scene, start, end, radius, color) -> None:
  import mujoco

  geom = scene.geoms[scene.ngeom]
  kind = mujoco.mjtGeom.mjGEOM_CAPSULE
  mujoco.mjv_initGeom(geom, kind, np.zeros(3), start, np.eye(3).ravel(), color)
  mujoco.mjv_connector(geom, kind, radius, start, end)
  scene.ngeom += 1


def _sphere(scene, point, radius, color, label: str = "") -> None:
  import mujoco

  geom = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(
    geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius), point, np.eye(3).ravel(), color
  )
  if label:
    geom.label = label
  scene.ngeom += 1


def _axes(scene, origin, rotation, length: float, radius: float) -> None:
  for axis, color in zip(rotation.T, AXIS_COLORS):
    _connector(scene, origin, origin + length * axis, radius, color)


def _draw_skeleton(scene, points, rotations, edges, color, radius, length, labels) -> None:
  for joint, name in enumerate(labels):
    _sphere(scene, points[joint], radius, color, name)
    _axes(scene, points[joint], rotations[joint], length, radius * 0.35)
  for parent, child in edges:
    if np.linalg.norm(points[child] - points[parent]) > 1e-7:
      _connector(scene, points[parent], points[child], radius * 0.45, color)


def _sparse_edges(names: list[str], parents: list[int | None], source_ids: list[int]) -> list[tuple[int, int]]:
  pairs = []
  wanted = set(source_ids)
  for joint in source_ids:
    parent = parents[joint]
    while parent is not None and parent not in wanted:
      parent = parents[parent]
    if parent is not None:
      pairs.append((source_ids.index(parent), source_ids.index(joint)))
  return pairs


def _draw(scene, data, frame: int, selected: int, nominal: bool, azimuth: float) -> None:
  scene.ngeom = 0
  theta = np.radians(azimuth)
  shift = np.array([np.sin(theta), -np.cos(theta), 0.0]) * 0.9
  bvh_frame = min(int(round(frame / data["fps"] * data["bvh_fps"])), len(data["bvh_pos"]) - 1)
  left = data["bvh_pos"][bvh_frame] - shift
  labels = [
    name if name in data["source"] or joint == selected else ""
    for joint, name in enumerate(data["names"])
  ]
  _draw_skeleton(
    scene, left, data["original"][bvh_frame], data["edges"],
    [0.15, 0.72, 0.72, 1.0], 0.012, 0.07, labels,
  )
  if nominal:
    right_pos = np.asarray(data["nominal_pos"], dtype=np.float64) + shift
    right_rot = Rotation.from_quat(data["nominal_quat"], scalar_first=True).as_matrix()
  else:
    right_pos = np.asarray(data["pos"][frame], dtype=np.float64)
    right_pos = right_pos.copy()
    right_pos[:, 2] += data["z_nom"]
    right_pos += shift
    right_rot = Rotation.from_quat(data["quat"][frame], scalar_first=True).as_matrix()
  sparse = _sparse_edges(data["names"], data["parents"], data["source_ids"])
  _draw_skeleton(
    scene, right_pos, right_rot, sparse,
    [0.97, 0.55, 0.16, 1.0], 0.016, 0.1, data["source"],
  )
  # Selected joint gets a longer triad so it can be matched to a robot body.
  _axes(scene, left[selected], data["original"][bvh_frame, selected], 0.16, 0.006)
  name = data["names"][selected]
  if name in data["source"]:
    index = data["source"].index(name)
    _axes(scene, right_pos[index], right_rot[index], 0.18, 0.007)


def _axis_text(rotation: np.ndarray) -> str:
  letters = "XYZ"
  return "  ".join(
    f"{letters[i]}=({rotation[0, i]:+.2f},{rotation[1, i]:+.2f},{rotation[2, i]:+.2f})"
    for i in range(3)
  )


def view(data) -> None:
  import mujoco
  import mujoco.viewer

  model = mujoco.MjModel.from_xml_string("""
    <mujoco model="link frames">
      <statistic center="0 0 0.9" extent="3"/>
      <asset>
        <texture name="grid" type="2d" builtin="checker" width="128" height="128"
                 rgb1=".19 .22 .24" rgb2=".26 .29 .31"/>
        <material name="floor" texture="grid" texrepeat="1 1" texuniform="true"/>
      </asset>
      <worldbody>
        <light pos="0 0 4" dir="0 0 -1" directional="true"/>
        <geom type="plane" size="0 0 .05" material="floor"/>
      </worldbody>
    </mujoco>""")
  state = mujoco.MjData(model)
  frame = selected = 0
  playing = False
  nominal = True
  count = len(data["pos"])
  timeline = 0.0
  keys: queue.SimpleQueue = queue.SimpleQueue()
  with mujoco.viewer.launch_passive(
    model, state, key_callback=keys.put, show_left_ui=False, show_right_ui=False
  ) as viewer:
    with viewer.lock():
      viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 90.0, -8.0, 4.2
      viewer.cam.lookat[:] = [0.0, 0.0, 0.85]
      viewer.opt.label = mujoco.mjtLabel.mjLABEL_GEOM
    previous = time.monotonic()
    while viewer.is_running():
      now = time.monotonic()
      if playing:
        timeline = (timeline + (now - previous) * data["fps"]) % count
      previous = now
      while not keys.empty():
        key = keys.get()
        if key == 32:
          playing = not playing
        elif key in (263, 262):
          playing = False
          timeline = float(np.clip(int(timeline) + (1 if key == 262 else -1), 0, count - 1))
        elif key in (ord("["), ord("]")):
          selected = (selected + (1 if key == ord("]") else -1)) % len(data["names"])
        elif key == ord("N"):
          nominal = not nominal
      frame = int(timeline)
      bvh_frame = min(int(round(frame / data["fps"] * data["bvh_fps"])), len(data["bvh_pos"]) - 1)
      with viewer.lock():
        _draw(viewer.user_scn, data, frame, selected, nominal, viewer.cam.azimuth)
      name = data["names"][selected]
      mode = "标称姿态 nominal_quat（配置机器人看这个）" if nominal else f"动作帧 {frame}/{count - 1}"
      extra = ""
      if name in data["source"]:
        index = data["source"].index(name)
        quat = data["nominal_quat"][index] if nominal else data["quat"][frame, index]
        extra = "\nNPZ " + _axis_text(Rotation.from_quat(quat, scalar_first=True).as_matrix())
      viewer.set_texts((
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "左：原始 BVH 关节轴（红 X，绿 Y，蓝 Z）\n"
        "右：转换后的 NPZ 刚体轴，世界系 Z 朝上、+X 朝前、+Y 朝左\n"
        f"{data['bvh']} → {data['npz']}\n{mode}\n"
        f"选中 {name}\nBVH " + _axis_text(data["original"][bvh_frame, selected]) + extra +
        "\n空格：播放动作    左右：逐帧    [ ]：切换关节    N：标称/动作",
        "",
      ))
      viewer.sync()
      time.sleep(1 / 60)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--bvh", type=Path, default=UNIFORM_RIG, help="原始 BVH，默认是训练用的标准骨架")
  parser.add_argument("--out", type=Path, help="转换后的 NPZ 路径")
  parser.add_argument("--rig", type=Path, default=UNIFORM_RIG, help="标定用的 Uniform 骨架")
  parser.add_argument("--fps", type=float, default=50.0)
  parser.add_argument("--no-view", action="store_true", help="只转换并打印轴向误差")
  args = parser.parse_args()
  out = args.out or (REPO / "outputs" / "link_frames" / f"{args.bvh.stem}.npz")
  try:
    data = prepare(args.bvh, out, args.rig, args.fps)
  except (OSError, ValueError, KeyError) as exc:
    parser.exit(1, f"{exc}\n")
  if not args.no_view:
    view(data)


if __name__ == "__main__":
  main()
