"""Compare a raw BONES-SEED BVH with an on-disk human training NPZ."""

from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

if __package__:
  from .convert_soma_bvh_to_npz import SOURCE_BODIES, UNIFORM_RIG, normalize_proportions, rig_calibration
  from .soma_bvh import BvhClip, parse_bvh, resample
else:
  from convert_soma_bvh_to_npz import SOURCE_BODIES, UNIFORM_RIG, normalize_proportions, rig_calibration
  from soma_bvh import BvhClip, parse_bvh, resample

REPO = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = REPO / "src/assets/motions/reactor/bones_seed/train/jog_sideway_045_stop_001__A021.npz"


def paired_bvh(npz: Path) -> Path:
  manifest = npz.parent / "clips.json"
  if manifest.is_file():
    for row in json.loads(manifest.read_text())["clips"]:
      if row["file"] == npz.name and "source" in row:
        for base in (npz.parent, REPO, REPO / ".tmp/soma_raw", Path("/home/kiku/data/RL/ASSETS/bones-seed")):
          path = base / row["source"]
          if path.is_file():
            return path
  raise ValueError("找不到配对的原始 BVH，请用 --bvh 指定文件。")


def aligned_fk(clip: BvhClip, yaw: float, wanted: list[int]):
  """Decode the raw channel rotations with SciPy, separately from the converter's FK."""
  n = len(clip.motion)
  positions = np.empty((n, len(clip.joints), 3))
  # A parent rotation is needed for its children; retain all world rotations.
  rotations = np.empty((n, len(clip.joints), 3, 3))
  offset = 0
  for j, joint in enumerate(clip.joints):
    block = clip.motion[:, offset:offset + len(joint.channels)]
    offset += len(joint.channels)
    translation = block[:, :3] if joint.num_pos_channels else joint.offset
    local = Rotation.from_euler("ZYX", block[:, -3:], degrees=True).as_matrix()
    parent = joint.parent
    if parent is None:
      rotations[:, j] = local
      positions[:, j] = translation
    else:
      rotations[:, j] = rotations[:, parent] @ local
      positions[:, j] = positions[:, parent] + (rotations[:, parent] @ translation[..., None])[..., 0]

  basis = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
  world = Rotation.from_euler("z", yaw).as_matrix() @ basis
  positions = positions @ world.T * .01
  positions[..., :2] -= positions[0, clip.joint_names.index("Hips"), :2].copy()
  matrices = world @ rotations[:, wanted] @ basis.T
  quat = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat(scalar_first=True)
  return positions, quat.reshape(n, len(wanted), 4)


def orientation_error(a, b):
  # q and -q describe the same orientation. Use chord length near zero.
  a = a / np.linalg.norm(a, axis=-1, keepdims=True)
  b = b / np.linalg.norm(b, axis=-1, keepdims=True)
  chord = np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
  return np.degrees(4 * np.arcsin(np.clip(chord / 2, 0., 1.)))


def expected_velocity(pos, quat, fps):
  lin = np.zeros_like(pos)
  ang = np.zeros_like(pos)
  if len(pos) > 1:
    lin[:-1] = np.diff(pos, axis=0) * fps
    rotations = Rotation.from_quat(quat.reshape(-1, 4), scalar_first=True)
    joints = pos.shape[1]
    delta = rotations[joints:] * rotations[:-joints].inv()
    ang[:-1] = delta.as_rotvec().reshape(ang[:-1].shape) * fps
    lin[-1], ang[-1] = lin[-2], ang[-2]
  return lin, ang


def read_npz(path: Path):
  with np.load(path, allow_pickle=False) as data:
    names = data["body_names"].tolist()
    if "Hips" not in names or "joint_pos" in data:
      raise ValueError("请传入训练用的 SOMA 人体 NPZ（body_names 含 Hips）。")
    if len(names) != len(set(names)) or not set(SOURCE_BODIES).issubset(names):
      raise ValueError(f"body_names 重复或缺少训练刚体：{set(SOURCE_BODIES) - set(names)}")
    ids = [names.index(name) for name in SOURCE_BODIES]
    arrays = []
    for key, width in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
      value = data[key]
      if (value.ndim != 3 or value.shape[1:] != (len(names), width) or not len(value)
          or (arrays and len(value) != len(arrays[0])) or not np.isfinite(value).all()):
        raise ValueError(f"{path.name}: {key} 的形状或数值异常")
      arrays.append(value[:, ids].astype(np.float64))
    order = data["quaternion_order"].item() if "quaternion_order" in data else "wxyz"
    if order == "xyzw":
      arrays[1] = arrays[1][..., [3, 0, 1, 2]]
    elif order != "wxyz":
      raise ValueError(f"未知四元数顺序 {order}")
    if np.any(np.linalg.norm(arrays[1], axis=-1) < 1e-8):
      raise ValueError("NPZ 包含零四元数")
    fps = float(data["fps"].item())
    if not np.isfinite(fps) or fps <= 0:
      raise ValueError("NPZ fps 必须是正数")
    return fps, arrays, float(data["z_nom"].item())


def compare(bvh: Path, npz: Path, rig: Path):
  fps, (pos, quat, lin, ang), z_nom = read_npz(npz)
  clip = parse_bvh(bvh)
  names = clip.joint_names
  indices = [names.index(j) for j in SOURCE_BODIES]
  n = len(clip.motion)
  expected = round((n - 1) * clip.frame_time * fps) + 1
  if len(pos) != expected:
    raise ValueError(f"时长不匹配：BVH {n} 帧 @ {clip.fps:g} Hz，应得到 {expected} 帧 @ {fps:g} Hz，NPZ 实有 {len(pos)} 帧。")
  yaw = rig_calibration(rig)["yaw"]
  raw, reference_quat = aligned_fk(clip, yaw, indices)
  raw = resample(raw, clip.fps, fps)
  reference = raw[:, indices]
  offsets = np.stack([j.offset for j in clip.joints])
  normalize_proportions(clip)
  proportional = not np.array_equal(offsets, np.stack([j.offset for j in clip.joints]))
  if proportional:
    normalized, reference_quat = aligned_fk(clip, yaw, indices)
    reference = resample(normalized[:, indices], clip.fps, fps)
  reference_quat = resample(reference_quat, clip.fps, fps, quaternion=True)
  ref_lin, ref_ang = expected_velocity(reference, reference_quat, fps)
  distances = np.linalg.norm(reference - pos, axis=-1) * 1000
  angles = orientation_error(reference_quat, quat)
  worst_frame, worst_joint = np.unravel_index(distances.argmax(), distances.shape)
  foot_ids = [SOURCE_BODIES.index(j) for j in ("LeftFoot", "RightFoot")]
  stats = dict(mean_mm=float(distances.mean()), max_mm=float(distances.max()),
               max_deg=float(angles.max()), worst_frame=int(worst_frame), worst_joint=SOURCE_BODIES[worst_joint],
               max_lin_ms=float(np.linalg.norm(lin - ref_lin, axis=-1).max()),
               max_ang_rads=float(np.linalg.norm(ang - ref_ang, axis=-1).max()),
               quat_norm_error=float(np.abs(np.linalg.norm(quat, axis=-1) - 1).max()),
               z_nom_error_mm=float(abs(z_nom + np.percentile(reference[:, foot_ids, 2], 2)) * 1000))
  print(f"{bvh.name} ({n} 帧 @ {clip.fps:g} Hz) → {npz.name} ({len(pos)} 帧 @ {fps:g} Hz)")
  print(f"17 刚体位置误差：平均 {stats['mean_mm']:.6g} mm，最大 {stats['max_mm']:.6g} mm"
        f"（第 {worst_frame} 帧，从 0 开始，{stats['worst_joint']}）")
  print(f"最大朝向误差 {stats['max_deg']:.6g}°；线速度 {stats['max_lin_ms']:.6g} m/s；"
        f"角速度 {stats['max_ang_rads']:.6g} rad/s；z_nom {stats['z_nom_error_mm']:.6g} mm；"
        f"四元数模长偏差 {stats['quat_norm_error']:.6g}")
  parents = [j.parent for j in clip.joints]
  raw_edges, sparse_edges = [], []
  for j in range(len(names)):
    parent = parents[j]
    if parent is not None and names[parent] != "Root":
      raw_edges.append([parent, j])
    if j in indices:
      while parent is not None and parent not in indices:
        parent = parents[parent]
      if parent is not None:
        sparse_edges.append([indices.index(parent), indices.index(j)])
  return dict(bvh=bvh.name, npz=npz.name, fps=fps, rawFPS=clip.fps, rawFrames=n, frames=len(pos),
              rawNames=names, names=SOURCE_BODIES, indices=indices, rawEdges=raw_edges, edges=sparse_edges,
              proportional=proportional, stats=stats, raw=raw, pos=pos, ref=reference, quat=quat,
              refQuat=reference_quat, mm=distances, deg=angles)


def draw_skeleton(scene, points, edges, visible, color, radius):
  import mujoco

  identity = np.eye(3).ravel()
  size = np.full(3, radius)
  for j in visible:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, points[j], identity, color)
    scene.ngeom += 1
  for a, b in edges:
    if np.linalg.norm(points[b] - points[a]) > 1e-7:
      connector(scene, points[a], points[b], radius * .65, color)


def connector(scene, start, end, radius, color):
  import mujoco

  geom = scene.geoms[scene.ngeom]
  kind = mujoco.mjtGeom.mjGEOM_CAPSULE
  mujoco.mjv_initGeom(geom, kind, np.zeros(3), start, np.eye(3).ravel(), color)
  mujoco.mjv_connector(geom, kind, radius, start, end)
  scene.ngeom += 1


def draw_frame(scene, motion, frame, joint, overlay, azimuth):
  scene.ngeom = 0
  # Offset along camera-right so orbiting keeps the two figures side by side.
  theta = np.radians(azimuth)
  shift = np.array([np.sin(theta), -np.cos(theta), 0.]) * (.0 if overlay else .78)
  if overlay:
    left = motion["ref"][frame]
    edges = motion["edges"]
    visible = range(len(left))
    selected = joint
  else:
    left = motion["raw"][frame] - shift
    edges = motion["rawEdges"]
    visible = [i for i, name in enumerate(motion["rawNames"]) if name != "Root"]
    selected = motion["indices"][joint]
  right = motion["pos"][frame] + shift
  draw_skeleton(scene, left, edges, visible, [.12, .70, .70, 1.], .019)
  draw_skeleton(scene, right, motion["edges"], range(len(right)), [.97, .55, .16, 1.], .012)
  for origin, q in ((left[selected], motion["refQuat"][frame, joint]),
                    (right[joint], motion["quat"][frame, joint])):
    axes = Rotation.from_quat(q, scalar_first=True).as_matrix().T
    for axis, color in zip(axes, ([1., .1, .1, 1.], [.1, .9, .1, 1.], [.2, .35, 1., 1.])):
      connector(scene, origin, origin + .12 * axis, .004, color)


def view_motion(motion):
  import mujoco
  import mujoco.viewer

  model = mujoco.MjModel.from_xml_string("""
    <mujoco model="BONES-SEED motion comparison">
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
  data = mujoco.MjData(model)
  keys = queue.SimpleQueue()
  names = motion["names"]
  frame = joint = 0
  playing, overlay = True, False
  fps, count = motion["fps"], motion["frames"]
  timeline = 0.
  roots = motion["raw"][:, motion["indices"][0], :2]
  last_root = roots[0]
  with mujoco.viewer.launch_passive(model, data, key_callback=keys.put,
                                    show_left_ui=False, show_right_ui=False) as viewer:
    with viewer.lock():
      viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 90., -10., 4.6
      viewer.cam.lookat[:] = [*last_root, .88]
    previous = time.monotonic()
    while viewer.is_running():
      now = time.monotonic()
      if playing:
        timeline = (timeline + (now - previous) * fps) % count
      previous = now
      while not keys.empty():
        key = keys.get()
        if key == 32:  # space
          playing = not playing
        elif key in (263, 262):  # GLFW left / right
          playing = False
          timeline = float(np.clip(int(timeline) + (1 if key == 262 else -1), 0, count - 1))
        elif key in (ord("["), ord("]")):
          joint = (joint + (1 if key == ord("]") else -1)) % len(names)
        elif key == ord("O"):
          overlay = not overlay
      frame = int(timeline)
      with viewer.lock():
        viewer.cam.lookat[:2] += roots[frame] - last_root
        last_root = roots[frame]
        draw_frame(viewer.user_scn, motion, frame, joint, overlay, viewer.cam.azimuth)
      mode = "Overlay: BVH reference / training NPZ" if overlay else "Left: raw BVH (teal)    Right: training NPZ (orange)"
      viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        f"{mode}\n{motion['npz']}\nFrame {frame}/{count-1}   {frame/fps:.2f} s   {fps:g} Hz"
                        f"\n{names[joint]}: {motion['mm'][frame, joint]:.4g} mm   "
                        f"{motion['deg'][frame, joint]:.4g} deg"
                        "\nSpace: play/pause    Left/Right: frame    [ / ]: joint    O: overlay", ""))
      viewer.sync()
      time.sleep(1 / 60)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ, help="训练用的人体源动作 NPZ")
  parser.add_argument("--bvh", type=Path, help="原始 BVH，默认按 clips.json 查找")
  parser.add_argument("--rig", type=Path, default=UNIFORM_RIG, help="转换时的标定 BVH")
  parser.add_argument("--check-only", action="store_true", help="只打印整段误差")
  args = parser.parse_args()
  try:
    payload = compare(args.bvh or paired_bvh(args.npz), args.npz, args.rig)
  except (OSError, ValueError, KeyError) as e:
    parser.exit(1, f"{e}\n")
  if not args.check_only:
    view_motion(payload)


if __name__ == "__main__":
  main()
