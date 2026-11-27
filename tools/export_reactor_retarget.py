"""Retarget BONES-SEED BVH or converted source NPZ clips by rolling out a frozen policy.

Each clip is written as a robot motion NPZ. Unless ``--no-video`` is set, a
MuJoCo video is written beside it: the original human skeleton on the left,
the retargeted robot on the right.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tarfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import islice
from shutil import copyfileobj
from pathlib import Path
from tempfile import TemporaryDirectory

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from src.assets.robots.pi_plus.pi_plus_constants import PI_PLUS_XML
from src.assets.robots.rp1.rp1_constants import RP1_24DOF_XML
from src.tasks.reactor.config.pi_plus.env_cfg import CALIBRATION as PI_PLUS_CALIBRATION
from src.tasks.reactor.config.pi_plus.env_cfg import pi_plus_reactor_env_cfg
from src.tasks.reactor.config.pi_plus.rl_cfg import pi_plus_reactor_runner_cfg
from src.tasks.reactor.config.reactor_env import MOTION_DIR
from src.tasks.reactor.config.rp1.env_cfg import CALIBRATION as RP1_CALIBRATION
from src.tasks.reactor.config.rp1.env_cfg import rp1_reactor_env_cfg
from src.tasks.reactor.config.rp1.rl_cfg import rp1_reactor_runner_cfg
from src.tasks.reactor.rl.runner import ReactorOnPolicyRunner
from tools.convert_soma_bvh_to_npz import SOURCE_BODIES, convert_clip
from tools.replay_npz import _add_floor, apply_frame, load_clip, load_robot_spec

# Sparse SOMA bones among the 17 tracked bodies, parent then child.
_HUMAN_BONES = (
  ("Hips", "Chest"), ("Chest", "Head"),
  ("Chest", "LeftShoulder"), ("LeftShoulder", "LeftArm"),
  ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
  ("Chest", "RightShoulder"), ("RightShoulder", "RightArm"),
  ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
  ("Hips", "LeftLeg"), ("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot"),
  ("Hips", "RightLeg"), ("RightLeg", "RightShin"), ("RightShin", "RightFoot"),
)
_HUMAN_COLOR = np.array([0.12, 0.72, 0.78, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class RobotExport:
  """One trained robot: play environment, runner hyperparameters, and MJCF."""

  name: str
  xml: Path
  env_cfg: Callable
  runner_cfg: Callable
  target_bodies: tuple[str, ...]


ROBOTS: dict[str, RobotExport] = {
  "pi_plus": RobotExport(
    "pi_plus",
    PI_PLUS_XML,
    pi_plus_reactor_env_cfg,
    pi_plus_reactor_runner_cfg,
    PI_PLUS_CALIBRATION.target_bodies,
  ),
  "rp1": RobotExport(
    "rp1",
    RP1_24DOF_XML,
    rp1_reactor_env_cfg,
    rp1_reactor_runner_cfg,
    RP1_CALIBRATION.target_bodies,
  ),
}


def select_robot(name: str | None, state: dict) -> RobotExport:
  """Use ``--robot`` when given, otherwise the checkpoint's trained body list."""
  if name is not None:
    if name not in ROBOTS:
      raise ValueError(f"Unknown robot {name!r}. Choices: {', '.join(ROBOTS)}")
    return ROBOTS[name]
  trained = tuple(state.get("retarget/target_bodies") or ())
  matches = [robot for robot in ROBOTS.values() if robot.target_bodies == trained]
  if len(matches) == 1:
    return matches[0]
  choices = ", ".join(ROBOTS)
  raise ValueError(f"Pass --robot {{{choices}}}. The checkpoint does not identify one robot.")


def source_files(src: Path, only: str, limit: int) -> list[Path]:
  paths = sorted(src.rglob("*")) if src.is_dir() else [src]
  paths = [p for p in paths if p.is_file() and p.suffix.lower() in {".bvh", ".npz"} and only in p.stem]
  if limit:
    paths = paths[:limit]
  if not paths:
    raise ValueError(f"No matching BVH or source NPZ clips in {src}")
  if len({p.stem for p in paths}) != len(paths):
    raise ValueError("Duplicate output names. Select one skeleton variant per export.")
  return paths


def _npz_frames(path: Path) -> int:
  """Frame count from the ``body_pos_w`` header, without reading the array."""
  import zipfile

  with zipfile.ZipFile(path) as archive:
    with archive.open("body_pos_w.npy") as raw:
      major, _minor = np.lib.format.read_magic(raw)
      if major == 1:
        shape, _, _ = np.lib.format.read_array_header_1_0(raw)
      else:
        shape, _, _ = np.lib.format.read_array_header_2_0(raw)
  return int(shape[0])


def source_batches(src: Path, only: str, limit: int, batch_size: int):
  """Read an archive sequentially, keeping at most one batch of raw BVHs."""
  if not src.name.endswith(".tar.gz"):
    paths = source_files(src, only, limit)
    if len(paths) > 1 and all(p.suffix.lower() == ".npz" for p in paths):
      # Similar lengths share a batch, so one long clip does not extend the rest.
      paths.sort(key=_npz_frames)
    for start in range(0, len(paths), batch_size):
      yield paths[start:start + batch_size]
    return
  seen = set()
  with tarfile.open(src, "r|gz") as archive:
    members = (m for m in archive if m.isfile() and m.name.lower().endswith(".bvh")
               and only in Path(m.name).stem)
    if limit:
      members = islice(members, limit)
    while True:
      with TemporaryDirectory() as tmp:
        batch = []
        for member in islice(members, batch_size):
          path = Path(tmp) / Path(member.name).name
          if path.stem in seen:
            raise ValueError(f"Duplicate clip name in archive: {path.stem}")
          seen.add(path.stem)
          with archive.extractfile(member) as source, path.open("wb") as target:
            copyfileobj(source, target)
          batch.append(path)
        if not batch:
          break
        yield batch
  if not seen:
    raise ValueError(f"No matching BVH in {src}")


def prepare(paths, temporary, preprocessing):
  result = []
  for path in paths:
    if path.suffix.lower() == ".bvh":
      data = convert_clip(path, 50.0, **preprocessing)
      path = temporary / (path.stem + ".npz")
      np.savez(path, **data)
    with np.load(path) as source:
      if ("source_yaw" not in source or
          not np.isclose(float(source["source_yaw"]), preprocessing["yaw"]) or
          not np.allclose(source["nominal_pos"], preprocessing["nominal_pos"]) or
          not np.allclose(source["nominal_quat"], preprocessing["nominal_quat"])):
        raise ValueError(f"{path} uses different source calibration. Supply the original BVH.")
    result.append(str(path))
  return result


def _bone_index() -> list[tuple[int, int]]:
  names = {name: i for i, name in enumerate(SOURCE_BODIES)}
  return [(names[parent], names[child]) for parent, child in _HUMAN_BONES]


def _draw_skeleton(scene, points: np.ndarray, edges: list[tuple[int, int]]) -> None:
  import mujoco

  identity = np.eye(3, dtype=np.float64).ravel()
  radius = 0.028
  size = np.full(3, radius)
  for point in points:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
      geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, point, identity, _HUMAN_COLOR
    )
    scene.ngeom += 1
  for parent, child in edges:
    start, end = points[parent], points[child]
    if np.linalg.norm(end - start) < 1e-6:
      continue
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
      geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), start, identity, _HUMAN_COLOR
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius * 0.65, start, end)
    scene.ngeom += 1


def _caption(image: np.ndarray) -> np.ndarray:
  from PIL import Image, ImageDraw, ImageFont

  picture = Image.fromarray(image)
  draw = ImageDraw.Draw(picture)
  try:
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
  except OSError:
    font = ImageFont.load_default()
  draw.text((18, 14), "Human motion", fill=(210, 245, 245), font=font)
  draw.text((image.shape[1] // 2 + 18, 14), "ReActor retarget", fill=(255, 255, 255), font=font)
  return np.asarray(picture)


def _follow(camera, root: np.ndarray, distance: float, height: float) -> None:
  camera.lookat[:] = (float(root[0]), float(root[1]), height)
  camera.distance = distance
  camera.azimuth = 135.0
  camera.elevation = -18.0


def render_comparison(source_pos: np.ndarray, robot_path: Path, out: Path, xml: Path) -> None:
  """Write a side-by-side MuJoCo video of one exported clip."""
  import imageio.v2 as imageio
  import mujoco

  edges = _bone_index()
  human_spec = mujoco.MjSpec()
  _add_floor(human_spec)
  human_spec.visual.global_.offwidth = 640
  human_spec.visual.global_.offheight = 540
  human_model = human_spec.compile()
  human_data = mujoco.MjData(human_model)
  mujoco.mj_forward(human_model, human_data)

  robot_spec = load_robot_spec(str(xml))
  _add_floor(robot_spec)
  robot_spec.visual.global_.offwidth = 640
  robot_spec.visual.global_.offheight = 540
  robot_model = robot_spec.compile()
  robot_data = mujoco.MjData(robot_model)
  robot_clip = load_clip(robot_path, robot_model)
  if robot_clip.n_frames != len(source_pos):
    raise RuntimeError(f"{robot_path.name} and its source motion have different lengths")

  human_renderer = mujoco.Renderer(human_model, height=540, width=640)
  robot_renderer = mujoco.Renderer(robot_model, height=540, width=640)
  human_cam, robot_cam = mujoco.MjvCamera(), mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(human_model, human_cam)
  mujoco.mjv_defaultFreeCamera(robot_model, robot_cam)
  out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(
    out, fps=robot_clip.fps, codec="libx264", quality=7, macro_block_size=1
  )
  try:
    for frame in range(robot_clip.n_frames):
      points = np.asarray(source_pos[frame], dtype=np.float64)
      _follow(human_cam, points[0], 3.8, 0.9)
      human_renderer.update_scene(human_data, camera=human_cam)
      _draw_skeleton(human_renderer.scene, points, edges)
      apply_frame(robot_model, robot_data, robot_clip, frame)
      _follow(robot_cam, robot_data.qpos[:3], 2.4, 0.5)
      robot_renderer.update_scene(robot_data, camera=robot_cam)
      writer.append_data(_caption(np.concatenate(
        [human_renderer.render(), robot_renderer.render()], axis=1
      )))
  finally:
    writer.close()
    human_renderer.close()
    robot_renderer.close()


@torch.inference_mode()
def rollout(paths, checkpoint, dst, device, seed, video: bool, robot: RobotExport):
  cfg = robot.env_cfg(play=True)
  cfg.seed = seed
  cfg.scene.num_envs = len(paths)
  cfg.events = {}
  cfg.rewards = {}
  cfg.terminations = {}
  motion = cfg.commands["motion"]
  motion.motion_files = tuple(paths)
  motion.collect_upper = False
  motion.sampling_mode = "start"
  agent_cfg = robot.runner_cfg()
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=device), clip_actions=agent_cfg.clip_actions)
  try:
    runner = ReactorOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), load_cfg={"actor": True})
    policy = runner.get_inference_policy(device)
    obs, _ = env.reset()
    cmd = env.unwrapped.command_manager.get_term("motion")
    while (cmd.psi < 1).any():
      obs, *_ = env.step(policy(obs))
    lengths = cmd.motion.lengths.tolist()
    buffers = {}
    for t in range(max(lengths)):
      # The policy sees g[t] before the step. Record the resulting state
      # against that same g[t], including the last frame of the clip.
      ref = cmd.reference()
      if video:
        source = cmd._source_frames(slice(None))[0]
      obs, *_ = env.step(policy(obs))
      entity = cmd.robot
      pos = cmd.robot_body_pos_w - env.unwrapped.scene.env_origins[:, None]
      quat = cmd.robot_body_quat_w
      positions = (pos - ref.pos).norm(dim=-1)
      angles = quat_error_magnitude(quat, ref.quat)
      local = quat_apply_inverse(quat[:, :1].expand_as(quat), pos - pos[:, :1])
      local_target = quat_apply_inverse(ref.quat[:, :1].expand_as(ref.quat), ref.pos - ref.pos[:, :1])
      limb_error = (local[:, 1:] - local_target[:, 1:]).norm(dim=-1).mean(1)
      wrench = env.unwrapped.action_manager.get_term("wrench").applied_wrench
      step = {
        "joint_pos": entity.data.joint_pos, "joint_vel": entity.data.joint_vel,
        "body_pos_w": pos, "body_quat_w": quat,
        "body_lin_vel_w": cmd.robot_body_lin_vel_w, "body_ang_vel_w": cmd.robot_body_ang_vel_w,
        "metrics": torch.stack([positions[:, 0], angles[:, 0], positions.mean(1),
                                angles.mean(1), wrench[:, :3].norm(dim=1),
                                wrench[:, 3:].norm(dim=1), limb_error], dim=1),
      }
      if video:
        step["source_pos"] = source
      for key, value in step.items():
        if key not in buffers:
          buffers[key] = value.new_empty((max(lengths), *value.shape))
        buffers[key][t].copy_(value)
    buffers = {key: value.cpu().numpy() for key, value in buffers.items()}
    summary = []
    for i, (name, length) in enumerate(zip(cmd.motion.names, lengths, strict=True)):
      data = {k: v[:length, i] for k, v in buffers.items() if k not in {"metrics", "source_pos"}}
      metrics = buffers["metrics"][:length, i]
      if not all(np.isfinite(a).all() for a in [*data.values(), metrics]):
        raise RuntimeError(f"Non-finite simulation state in {name}")
      root_pos, root_rot, body_pos, body_rot, force, torque, limb_error = metrics.mean(axis=0).tolist()
      failures = np.flatnonzero((metrics[:, 0] > 1.0) | (metrics[:, 1] > np.pi / 4))
      directory = dst / "failed" if failures.size else dst
      directory.mkdir(parents=True, exist_ok=True)
      out = directory / f"{name}.npz"
      np.savez_compressed(out, fps=np.array([cmd.motion.fps], np.float32),
                          joint_names=entity.joint_names, body_names=cmd.cfg.target_bodies,
                          quaternion_order="wxyz", anchor_body=cmd.cfg.target_bodies[0], **data)
      video_path = None
      if video:
        video_path = dst / "videos" / f"{name}.mp4"
        render_comparison(buffers["source_pos"][:length, i], out, video_path, robot.xml)
      row = dict(file=str(out.relative_to(dst)), frames=length, success=not failures.size,
                 video=None if video_path is None else str(video_path.relative_to(dst)),
                 first_failed_frame=int(failures[0]) if failures.size else None,
                 root_position_error_m=root_pos, root_orientation_error_deg=np.rad2deg(root_rot),
                 body_position_error_m=body_pos, body_orientation_error_deg=np.rad2deg(body_rot),
                 root_relative_body_error_m=limb_error, residual_force_N=force, residual_torque_Nm=torque)
      print(json.dumps(row), flush=True)
      summary.append(row)
    return summary
  finally:
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument(
    "--robot",
    choices=tuple(ROBOTS),
    help="Robot to roll out. Default: the body list stored in the checkpoint.",
  )
  parser.add_argument("--src", type=Path, default=MOTION_DIR, help="BVH, source NPZ, directory, or SOMA tar.gz archive")
  parser.add_argument("--dst", type=Path, required=True)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--only", default="")
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--no-video", action="store_true",
                      help="Skip the side-by-side MuJoCo video and only write motion files")
  parser.add_argument("--resume", action="store_true",
                      help="Skip clips whose NPZ is already in the output directory")
  args = parser.parse_args()
  if args.batch_size < 1 or args.limit < 0:
    parser.error("batch-size must be positive and limit non-negative")
  state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["infos"]["reactor"]
  if state.get("version") != 2 or state["source_preprocessing"]["yaw"] is None:
    parser.error("Use a motion-conditioned checkpoint trained with the BONES-SEED uniform calibration")
  try:
    robot = select_robot(args.robot, state)
  except ValueError as exc:
    parser.error(str(exc))
  print(f"Export robot: {robot.name}")
  if args.dst.exists() and any(args.dst.iterdir()) and not args.resume:
    parser.error("Choose an empty output directory to keep successful and failed runs distinct")
  args.dst.mkdir(parents=True, exist_ok=True)
  done = {
    path.name
    for folder in (args.dst, args.dst / "failed") if folder.is_dir()
    for path in folder.glob("*.npz")
  }
  summary = []
  report_path = args.dst / "clips.json"
  if args.resume and report_path.is_file():
    summary = [row for row in json.loads(report_path.read_text())["clips"] if Path(row["file"]).name in done]
  for paths in source_batches(args.src, args.only, args.limit, args.batch_size):
    paths = [path for path in paths if f"{path.stem}.npz" not in done]
    if not paths:
      continue
    with TemporaryDirectory() as tmp:
      batch = prepare(paths, Path(tmp), state["source_preprocessing"])
      summary += rollout(
        batch, args.checkpoint, args.dst, args.device, args.seed,
        video=not args.no_video, robot=robot,
      )
    done.update(Path(row["file"]).name for row in summary)
    report = dict(
      checkpoint=str(args.checkpoint.resolve()), robot=robot.name, seed=args.seed, clips=summary,
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  print(f"Completed {len(summary)} clips, {sum(r['success'] for r in summary)} passed root tracking thresholds.")


if __name__ == "__main__":
  main()
