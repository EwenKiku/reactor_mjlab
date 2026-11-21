"""Play a retargeted clip kinematically in the MuJoCo viewer.

The clip is posed kinematically (no physics): the root body pose and
``joint_pos`` are written into ``qpos`` at the file's ``fps``. Joint columns
are matched by ``joint_names`` and the root body by ``body_names``. Quaternions
are converted to MuJoCo's ``wxyz`` using the ``quaternion_order`` field.

Example:
  python tools/replay_npz.py \
    --motion src/assets/motions/reactor/pi_plus/walk_forward_professional_impro_001__A002_M.npz \
    --xml src/assets/robots/pi_plus/xmls/pi_plus.xml
  python tools/replay_npz.py \
    --motion src/assets/motions/reactor/pi_plus \
    --xml src/assets/robots/pi_plus/xmls/pi_plus.xml
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

# GLFW key codes used by the passive viewer callback.
_KEY_SPACE = 32
_KEY_LEFT = 263
_KEY_RIGHT = 262
_KEY_UP = 265
_KEY_DOWN = 264
_KEY_R = 82

_QUAT_TO_WXYZ = {"wxyz": (0, 1, 2, 3), "xyzw": (3, 0, 1, 2)}


@dataclass
class PlayNpzConfig:
  motion: str
  """A ``.npz`` file or a directory of clips."""
  xml: str
  """MJCF path of the robot."""
  speed: float = 1.0
  """Playback rate relative to the clip fps."""
  loop: bool = True
  """Repeat the current clip. A directory still advances with the arrow keys."""


def load_robot_spec(xml: str) -> mujoco.MjSpec:
  path = Path(xml).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  return mujoco.MjSpec.from_file(str(path))


def _add_floor(spec: mujoco.MjSpec) -> None:
  """Add a lit checkerboard plane when the model has no ground."""
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
      return
  spec.add_texture(
    name="ground_checker",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    mark=mujoco.mjtMark.mjMARK_EDGE,
    rgb1=[0.88, 0.90, 0.92],
    rgb2=[0.42, 0.46, 0.50],
    markrgb=[0.96, 0.96, 0.96],
    width=300,
    height=300,
  )
  material = spec.add_material(
    name="ground_checker",
    texuniform=True,
    texrepeat=[2.0, 2.0],
    reflectance=0.05,
  )
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB.value] = "ground_checker"
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0.0, 0.0, 0.05],
    material="ground_checker",
  )
  spec.worldbody.add_light(
    name="floor_light",
    pos=[0.0, 0.0, 6.0],
    dir=[0.0, 0.0, -1.0],
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    castshadow=False,
    diffuse=[0.8, 0.8, 0.8],
    ambient=[0.45, 0.45, 0.45],
  )
  spec.visual.headlight.ambient[:] = [0.35, 0.35, 0.35]
  spec.visual.headlight.diffuse[:] = [0.7, 0.7, 0.7]
  spec.visual.headlight.active = 1


def _to_wxyz(quat: np.ndarray, order: str) -> np.ndarray:
  perm = _QUAT_TO_WXYZ.get(order)
  if perm is None:
    raise ValueError(f"Unsupported quaternion order {order!r}.")
  return quat[..., list(perm)]


def _names(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> list[str] | None:
  for key in keys:
    if key in data.files:
      return [str(name) for name in data[key]]
  return None


@dataclass
class Clip:
  path: Path
  fps: float
  root_pos: np.ndarray
  root_rot: np.ndarray
  dof_pos: np.ndarray
  joint_ids: np.ndarray
  """Column in ``dof_pos`` for each hinge, in model joint order."""

  @property
  def n_frames(self) -> int:
    return int(self.root_pos.shape[0])


def _free_joint(model: mujoco.MjModel) -> tuple[int, str]:
  """``(qpos address, body name)`` of the single free joint."""
  free = [
    (int(model.jnt_qposadr[jid]), int(model.jnt_bodyid[jid]))
    for jid in range(model.njnt)
    if int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_FREE
  ]
  if len(free) != 1:
    raise ValueError(f"Expected one free joint, found {len(free)}.")
  adr, body_id = free[0]
  name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
  if name is None:
    raise ValueError("The free joint's body has no name.")
  return adr, name


def _hinge_joints(model: mujoco.MjModel) -> list[tuple[str, int]]:
  hinges: list[tuple[str, int]] = []
  for jid in range(model.njnt):
    if int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if name is None:
      continue
    if int(model.jnt_type[jid]) != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError(f"Joint {name!r} is not a hinge. This player only poses hinges.")
    hinges.append((name, int(model.jnt_qposadr[jid])))
  return hinges


def load_clip(path: Path, model: mujoco.MjModel) -> Clip:
  with np.load(path) as raw:
    for key in ("fps", "joint_pos", "body_pos_w", "body_quat_w"):
      if key not in raw.files:
        raise ValueError(f"{path.name} is missing {key!r}.")
    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    body_pos = np.asarray(raw["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(raw["body_quat_w"], dtype=np.float64)
    dof_pos = np.asarray(raw["joint_pos"], dtype=np.float64)
    order = "wxyz"
    if "quaternion_order" in raw.files:
      order = str(np.asarray(raw["quaternion_order"]).item())
    body_names = _names(raw, ("body_names", "body_order"))
    joint_names = _names(raw, ("joint_names", "joint_order"))

  if body_pos.shape[0] != dof_pos.shape[0] or body_pos.shape[0] < 2:
    raise ValueError(f"{path.name} needs at least 2 aligned body and joint frames.")
  if body_quat.shape[:2] != body_pos.shape[:2]:
    raise ValueError(f"{path.name} body_quat_w does not match body_pos_w.")

  _, root_body = _free_joint(model)
  if body_names is not None:
    if root_body not in body_names:
      raise ValueError(f"{path.name} has no pose for the root body {root_body!r}.")
    root = body_names.index(root_body)
  else:
    root = 0
  root_pos = body_pos[:, root]
  root_rot = _to_wxyz(body_quat[:, root], order)
  root_rot = root_rot / np.linalg.norm(root_rot, axis=-1, keepdims=True)

  hinges = _hinge_joints(model)
  if joint_names is not None:
    # A clip may omit hinges it does not drive (the head joints are spring-held
    # at zero on this robot).  Append one neutral column per missing hinge
    # instead of failing the whole playback.
    existing = {name: i for i, name in enumerate(joint_names)}
    missing = [name for name, _ in hinges if name not in existing]
    if missing:
      print(
        f"[WARN] {path.name}: no data for {missing}; holding their neutral pose."
      )
    extra: list[np.ndarray] = []
    joint_ids = np.empty(len(hinges), dtype=np.int64)
    for i, (name, adr) in enumerate(hinges):
      if name in existing:
        joint_ids[i] = existing[name]
      else:
        joint_ids[i] = dof_pos.shape[1] + len(extra)
        extra.append(np.full((dof_pos.shape[0], 1), model.qpos0[adr]))
    if extra:
      dof_pos = np.concatenate([dof_pos, *extra], axis=1)
  elif dof_pos.shape[1] == len(hinges):
    joint_ids = np.arange(len(hinges), dtype=np.int64)
  else:
    raise ValueError(
      f"{path.name} has {dof_pos.shape[1]} dofs and no joint_names. "
      f"The model has {len(hinges)} hinges."
    )
  return Clip(path, fps, root_pos, root_rot, dof_pos, joint_ids)


def list_clips(motion: Path) -> list[Path]:
  if motion.is_file():
    if motion.suffix != ".npz":
      raise ValueError(f"Expected a .npz file, got {motion}")
    return [motion]
  if motion.is_dir():
    clips = sorted(motion.glob("*.npz"))
    if not clips:
      raise FileNotFoundError(f"No .npz files in {motion}")
    return clips
  raise FileNotFoundError(motion)


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, clip: Clip, frame: int) -> None:
  hinges = _hinge_joints(model)
  free, _ = _free_joint(model)
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  data.qpos[free : free + 3] = clip.root_pos[frame]
  data.qpos[free + 3 : free + 7] = clip.root_rot[frame]
  dof = clip.dof_pos[frame]
  for column, (_, adr) in zip(clip.joint_ids, hinges, strict=True):
    data.qpos[adr] = dof[int(column)]
  mujoco.mj_forward(model, data)


class _Playback:
  def __init__(self, n_clips: int, speed: float) -> None:
    self.paused = False
    self.speed = speed
    self.clip_index = 0
    self.frame = 0
    self.restart = False
    self.n_clips = n_clips

  def on_key(self, key: int) -> None:
    if key == _KEY_SPACE:
      self.paused = not self.paused
    elif key == _KEY_RIGHT:
      self.clip_index = (self.clip_index + 1) % self.n_clips
      self.frame = 0
      self.restart = True
    elif key == _KEY_LEFT:
      self.clip_index = (self.clip_index - 1) % self.n_clips
      self.frame = 0
      self.restart = True
    elif key == _KEY_UP:
      self.speed = min(self.speed * 1.25, 8.0)
    elif key == _KEY_DOWN:
      self.speed = max(self.speed / 1.25, 0.125)
    elif key == _KEY_R:
      self.frame = 0
      self.restart = True


def main() -> None:
  cfg = tyro.cli(PlayNpzConfig)
  motion = Path(cfg.motion).expanduser().resolve()
  paths = list_clips(motion)
  spec = load_robot_spec(cfg.xml)
  _add_floor(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  clips = [load_clip(path, model) for path in paths]
  free_adr, _ = _free_joint(model)
  play = _Playback(len(clips), cfg.speed)

  print(f"[INFO] xml={cfg.xml}  clips={len(clips)}")
  print("space pause | left/right clip | up/down speed | r restart | close window to quit")

  with mujoco.viewer.launch_passive(model, data, key_callback=play.on_key) as viewer:
    viewer.cam.distance = 3.2
    viewer.cam.elevation = -15.0
    viewer.cam.azimuth = 135.0
    viewer.cam.lookat[:] = clips[0].root_pos[0]
    last = time.perf_counter()
    accum = 0.0
    shown = -1
    while viewer.is_running():
      now = time.perf_counter()
      dt = now - last
      last = now
      clip = clips[play.clip_index]
      if shown != play.clip_index or play.restart:
        print(
          f"[clip] {clip.path.name}  {clip.n_frames} frames @ {clip.fps:g} Hz  "
          f"speed={play.speed:.2f}x"
        )
        shown = play.clip_index
        play.restart = False
        accum = 0.0
      if not play.paused:
        accum += dt * play.speed
        step = 1.0 / clip.fps
        while accum >= step:
          accum -= step
          play.frame += 1
          if play.frame >= clip.n_frames:
            if cfg.loop or len(clips) == 1:
              play.frame = 0
            else:
              play.clip_index = (play.clip_index + 1) % len(clips)
              play.frame = 0
              play.restart = True
              break
      apply_frame(
        model,
        data,
        clips[play.clip_index],
        min(play.frame, clips[play.clip_index].n_frames - 1),
      )
      viewer.cam.lookat[:] = data.qpos[free_adr : free_adr + 3]
      viewer.sync()
      time.sleep(0.001)


if __name__ == "__main__":
  main()
