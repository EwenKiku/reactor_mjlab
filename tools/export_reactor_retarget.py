"""Retarget BONES-SEED BVH or converted source NPZ clips by rolling out a frozen policy."""

from __future__ import annotations

import argparse
import json
import tarfile
from itertools import islice
from shutil import copyfileobj
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from src.tasks.reactor.config.pi_plus.env_cfg import MOTION_DIR, pi_plus_reactor_env_cfg
from src.tasks.reactor.config.pi_plus.rl_cfg import pi_plus_reactor_runner_cfg
from src.tasks.reactor.rl.runner import ReactorOnPolicyRunner
from tools.convert_soma_bvh_to_npz import convert_clip


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


def source_batches(src: Path, only: str, limit: int, batch_size: int):
  """Read an archive sequentially, keeping at most one batch of raw BVHs."""
  if not src.name.endswith(".tar.gz"):
    paths = source_files(src, only, limit)
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


@torch.inference_mode()
def rollout(paths, checkpoint, dst, device, seed):
  cfg = pi_plus_reactor_env_cfg(play=True)
  cfg.seed = seed
  cfg.scene.num_envs = len(paths)
  cfg.events = {}
  cfg.rewards = {}
  cfg.terminations = {}
  motion = cfg.commands["motion"]
  motion.motion_files = tuple(paths)
  motion.collect_upper = False
  motion.sampling_mode = "start"
  agent_cfg = pi_plus_reactor_runner_cfg()
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
      obs, *_ = env.step(policy(obs))
      robot = cmd.robot
      pos = cmd.robot_body_pos_w - env.unwrapped.scene.env_origins[:, None]
      quat = cmd.robot_body_quat_w
      positions = (pos - ref.pos).norm(dim=-1)
      angles = quat_error_magnitude(quat, ref.quat)
      local = quat_apply_inverse(quat[:, :1].expand_as(quat), pos - pos[:, :1])
      local_target = quat_apply_inverse(ref.quat[:, :1].expand_as(ref.quat), ref.pos - ref.pos[:, :1])
      limb_error = (local[:, 1:] - local_target[:, 1:]).norm(dim=-1).mean(1)
      wrench = env.unwrapped.action_manager.get_term("wrench").applied_wrench
      step = {
        "joint_pos": robot.data.joint_pos, "joint_vel": robot.data.joint_vel,
        "body_pos_w": pos, "body_quat_w": quat,
        "body_lin_vel_w": cmd.robot_body_lin_vel_w, "body_ang_vel_w": cmd.robot_body_ang_vel_w,
        "metrics": torch.stack([positions[:, 0], angles[:, 0], positions.mean(1),
                                angles.mean(1), wrench[:, :3].norm(dim=1),
                                wrench[:, 3:].norm(dim=1), limb_error], dim=1),
      }
      for key, value in step.items():
        if key not in buffers:
          buffers[key] = value.new_empty((max(lengths), *value.shape))
        buffers[key][t].copy_(value)
    buffers = {key: value.cpu().numpy() for key, value in buffers.items()}
    summary = []
    for i, (name, length) in enumerate(zip(cmd.motion.names, lengths, strict=True)):
      data = {k: v[:length, i] for k, v in buffers.items() if k != "metrics"}
      metrics = buffers["metrics"][:length, i]
      if not all(np.isfinite(a).all() for a in [*data.values(), metrics]):
        raise RuntimeError(f"Non-finite simulation state in {name}")
      root_pos, root_rot, body_pos, body_rot, force, torque, limb_error = metrics.mean(axis=0).tolist()
      failures = np.flatnonzero((metrics[:, 0] > 1.0) | (metrics[:, 1] > np.pi / 4))
      directory = dst / "failed" if failures.size else dst
      directory.mkdir(parents=True, exist_ok=True)
      out = directory / f"{name}.npz"
      np.savez_compressed(out, fps=np.array([cmd.motion.fps], np.float32),
                          joint_names=robot.joint_names, body_names=cmd.cfg.target_bodies,
                          quaternion_order="wxyz", anchor_body=cmd.cfg.target_bodies[0], **data)
      row = dict(file=str(out.relative_to(dst)), frames=length, success=not failures.size,
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
  parser.add_argument("--src", type=Path, default=MOTION_DIR, help="BVH, source NPZ, directory, or SOMA tar.gz archive")
  parser.add_argument("--dst", type=Path, required=True)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--only", default="")
  parser.add_argument("--limit", type=int, default=0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.batch_size < 1 or args.limit < 0:
    parser.error("batch-size must be positive and limit non-negative")
  state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["infos"]["reactor"]
  if state.get("version") != 2 or state["source_preprocessing"]["yaw"] is None:
    parser.error("Use a motion-conditioned checkpoint trained with the BONES-SEED uniform calibration")
  if args.dst.exists() and any(args.dst.iterdir()):
    parser.error("Choose an empty output directory to keep successful and failed runs distinct")
  args.dst.mkdir(parents=True, exist_ok=True)
  summary = []
  for paths in source_batches(args.src, args.only, args.limit, args.batch_size):
    with TemporaryDirectory() as tmp:
      batch = prepare(paths, Path(tmp), state["source_preprocessing"])
      summary += rollout(batch, args.checkpoint, args.dst, args.device, args.seed)
    report = dict(checkpoint=str(args.checkpoint.resolve()), seed=args.seed, clips=summary)
    (args.dst / "clips.json").write_text(json.dumps(report, indent=2) + "\n")
  print(f"Completed {len(summary)} clips, {sum(r['success'] for r in summary)} passed root tracking thresholds.")


if __name__ == "__main__":
  main()
