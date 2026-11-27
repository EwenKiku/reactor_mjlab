"""ReActor retargeting parameterization and upper-level loss.

Implements the parameterization of Müller et al. 2026 (ACM TOG) "ReActor:
Reinforcement Learning for Physics-Aware Motion Retargeting":

* Nominal calibration (Eq. 8-9), computed once in numpy from a nominal source
  pose and a nominal robot pose.
* The parameterized mapping from source frames ``m`` to target reference ``g``
  (Eq. 10-13), differentiable in the retargeting parameters ``p``.
* The upper-level loss (Eq. 15-17).

The source is a SOMA skeleton driven by BONES-SEED BVH motion; the target is a
robot whose body names come from its MJCF.  Only a sparse set of semantic
body pairs is required.

Everything the policy sees as the *reference* is ``g``; the upper level compares
``g`` to the simulated state ``s`` and pushes ``p`` along a simplified gradient
(Eq. 6-7), i.e. ``d/dL = (1 - alpha) * dL/dg * dg/dp`` with ``s`` detached.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_mul,
)

if TYPE_CHECKING:
  from collections.abc import Sequence


def quat_from_rotvec(
  rotvec: torch.Tensor, eps: float = 1.0e-9
) -> torch.Tensor:
  """Rodrigues exponential map, ``(..., 3)`` rotation vectors -> ``wxyz`` quats.

  Differentiable in ``rotvec``, which is what Eq. (11) needs for ``p_ori``.
  """
  angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True).clamp(min=eps)
  half = 0.5 * angle
  scale = torch.sin(half) / angle
  return torch.cat([torch.cos(half), rotvec * scale], dim=-1)


def quat_orthonormalize(quat: torch.Tensor) -> torch.Tensor:
  return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1.0e-9)


def yaw_only(quat: torch.Tensor) -> torch.Tensor:
  """Project a ``wxyz`` quaternion onto its yaw (world-z) component."""
  w, x, y, z = quat.unbind(-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  out = torch.zeros_like(quat)
  out[..., 0] = torch.cos(0.5 * yaw)
  out[..., 3] = torch.sin(0.5 * yaw)
  return out


##
# Nominal calibration.
##


@dataclasses.dataclass
class Correspondence:
  """One semantic source/target rigid-body pair (paper Sec. 5)."""

  source: str
  target: str
  is_root: bool = False

  @property
  def name(self) -> str:
    return f"{self.source}->{self.target}"


@dataclasses.dataclass
class NominalCalibration:
  """Constants extracted from the nominal source and target configurations.

  ``x_nom[b]`` and ``R_nom[b]`` are Eq. (8)-(9), expressed in the local frame of
  the ``b``-th source body.  ``scale`` is the global root-height ratio.
  """

  scale: float
  x_nom: np.ndarray  # (num_corr, 3)
  R_nom: np.ndarray  # (num_corr, 3, 3)
  source_height: float
  target_height: float

  @classmethod
  def build(
    cls,
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    target_pos: np.ndarray,
    npz_to_robot: np.ndarray,
    source_height: float,
    target_height: float,
  ) -> NominalCalibration:
    """Calibrate positions from the nominal poses and orientations from ``npz_to_robot``.

    ``npz_to_robot`` is ``(B, 3, 3)`` with ``v_robot = R @ v_npz``. The stored
    ``R_nom`` is its transpose, so ``R_robot = R_source @ R_nom`` still holds.
    Quaternions are ``wxyz``.
    """
    scale = float(target_height / max(source_height, 1.0e-6))
    from mjlab.utils.lab_api.math import matrix_from_quat as _m

    rotation = np.asarray(npz_to_robot, dtype=np.float64)
    if rotation.shape != (source_pos.shape[0], 3, 3):
      raise ValueError(
        f"npz_to_robot shape {rotation.shape} does not match {source_pos.shape[0]} bodies"
      )
    r_source = _m(torch.as_tensor(source_quat, dtype=torch.float64)).numpy()
    x_nom = np.einsum(
      "bij,bj->bi", np.transpose(r_source, (0, 2, 1)), target_pos - scale * source_pos
    )
    r_nom = np.transpose(rotation, (0, 2, 1))
    return cls(
      scale=scale,
      x_nom=x_nom,
      R_nom=r_nom,
      source_height=float(source_height),
      target_height=float(target_height),
    )


##
# Parameterized mapping, Eq. (10)-(13).
##


@dataclasses.dataclass
class RetargetParams:
  """Learnable retargeting parameters ``p = {p_pos, p_ori, p_z}``."""

  p_pos: torch.Tensor  # (num_corr, 3)
  p_ori: torch.Tensor  # (num_corr, 3)
  p_z: torch.Tensor  # (num_motions,), one learnable vertical offset per clip

  @classmethod
  def zeros(
    cls, num_corr: int, num_motions: int = 1, device: str | torch.device = "cpu"
  ) -> RetargetParams:
    return cls(
      p_pos=torch.zeros(num_corr, 3, device=device),
      p_ori=torch.zeros(num_corr, 3, device=device),
      p_z=torch.zeros(num_motions, device=device),
    )

  def as_list(self) -> list[torch.Tensor]:
    return [self.p_pos, self.p_ori, self.p_z]

  def clone(self) -> RetargetParams:
    return RetargetParams(self.p_pos.clone(), self.p_ori.clone(), self.p_z.clone())

  def detach(self) -> RetargetParams:
    return RetargetParams(self.p_pos.detach(), self.p_ori.detach(), self.p_z.detach())

  def assign_from(self, other: RetargetParams) -> None:
    """Copy values in place so existing graph/optimizer references stay valid."""
    self.p_pos.copy_(other.p_pos)
    self.p_ori.copy_(other.p_ori)
    self.p_z.copy_(other.p_z)

  def add_scaled_(self, other: RetargetParams, factor: float) -> None:
    self.p_pos.add_(other.p_pos, alpha=factor)
    self.p_ori.add_(other.p_ori, alpha=factor)
    self.p_z.add_(other.p_z, alpha=factor)

  def norms(self) -> dict[str, float]:
    return {
      "retarget/pos_norm": float(self.p_pos.norm(dim=-1).mean()),
      "retarget/ori_norm": float(self.p_ori.norm(dim=-1).mean()),
      "retarget/z_shift": float(self.p_z.mean()),
    }

  @staticmethod
  def project_(
    params: RetargetParams,
    delta_pos: float,
    delta_ori: float,
    delta_z: float,
  ) -> None:
    """Euclidean projection onto the convex set ``P`` of Eq. (14)."""
    with torch.no_grad():
      for value, bound in ((params.p_pos, delta_pos), (params.p_ori, delta_ori)):
        norm = value.norm(dim=-1, keepdim=True)
        scale = torch.where(norm > bound, bound / norm.clamp(min=1.0e-9), 1.0)
        value.mul_(scale)
      params.p_z.clamp_(min=-delta_z, max=delta_z)


class RetargetMapping:
  """Constant nominal data + the differentiable ``m -> g`` map.

  Tensors are ordered ``(num_corr, ...)`` on a fixed device.  ``source_*``
  inputs are per-environment ``(num_envs, num_corr, ...)``.
  """

  def __init__(
    self,
    calibration: NominalCalibration,
    z_nom: torch.Tensor,
    device: str | torch.device = "cpu",
  ) -> None:
    self.num_corr = calibration.x_nom.shape[0]
    self.scale = float(calibration.scale)
    self.x_nom = torch.as_tensor(calibration.x_nom, dtype=torch.float32, device=device)
    self.R_nom = torch.as_tensor(
      _matrix_to_wxyz(calibration.R_nom), dtype=torch.float32, device=device
    )
    # Per-clip nominal vertical offset (paper: z_nom, precomputed per motion).
    self.z_nom = z_nom.to(device=device, dtype=torch.float32)
    self.device = device

  def __call__(
    self,
    params: RetargetParams,
    source_pos: torch.Tensor,
    source_quat: torch.Tensor,
    source_lin_vel: torch.Tensor,
    source_ang_vel: torch.Tensor,
    z_shift: torch.Tensor,
    psi: torch.Tensor | None = None,
  ) -> RetargetedState:
    """Apply Eq. (10)-(13).

    Args:
      params: retargeting parameters, differentiable.
      source_pos/source_quat/source_lin_vel/source_ang_vel: ``(N, B, 3|4)``
        source body frames in the world frame, ``quat`` in ``wxyz``.
      z_shift: ``(N,)`` total per-environment vertical shift ``z_nom + p_z``.
        The command indexes ``p_z`` by each env's clip, so this term already
        carries the gradient back to the per-clip parameters.
      psi: ``(N,)`` retargeting phase; unused here, kept for symmetry.
    """
    # Eq. (10) local offset: R_nom p_pos + x_nom
    num_env, num_corr = source_pos.shape[:2]
    p_pos = params.p_pos.unsqueeze(0).expand(num_env, -1, -1)
    offset_local = (
      quat_apply(self.R_nom.unsqueeze(0).expand(num_env, -1, -1), p_pos)
      + self.x_nom.unsqueeze(0)
    )
    offset_world = quat_apply(source_quat, offset_local)

    # (N,) -> (N, 1, 3): the shift is the same for every body of that env.
    shift = z_shift.unsqueeze(-1).unsqueeze(-1)
    ground = torch.cat([torch.zeros_like(shift), torch.zeros_like(shift), shift], dim=-1)

    pos_g = offset_world + self.scale * source_pos + ground
    # Eq. (11)
    ori_g = quat_mul(
      quat_mul(source_quat, self.R_nom.unsqueeze(0).expand_as(source_quat)),
      quat_from_rotvec(params.p_ori.unsqueeze(0).expand(source_quat.shape[:-1] + (3,))),
    )
    # Eq. (12)-(13)
    lin_vel_g = torch.cross(source_ang_vel, offset_world, dim=-1) + self.scale * source_lin_vel
    ang_vel_g = source_ang_vel
    return RetargetedState(
      pos=pos_g, quat=ori_g, lin_vel=lin_vel_g, ang_vel=ang_vel_g
    )


@dataclasses.dataclass
class RetargetedState:
  """The parameterized reference ``g_t`` in the world frame."""

  pos: torch.Tensor  # (N, B, 3)
  quat: torch.Tensor  # (N, B, 4), wxyz
  lin_vel: torch.Tensor
  ang_vel: torch.Tensor


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
  from mjlab.utils.lab_api.math import quat_from_matrix as _q

  return _q(torch.as_tensor(matrix, dtype=torch.float64)).numpy()


##
# Upper-level loss, Eq. (15)-(17).
##


@dataclasses.dataclass
class LossWeights:
  pos: float = 10.0
  ori: float = 1.0
  lin_vel: float = 0.0
  ang_vel: float = 0.0


def retargeting_loss(
  reference: RetargetedState,
  simulated_pos: torch.Tensor,
  simulated_quat: torch.Tensor,
  simulated_lin_vel: torch.Tensor,
  simulated_ang_vel: torch.Tensor,
  weights: LossWeights,
  root_index: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Per-environment upper-level loss ``ell(g_t - s_t)``.

  ``simulated_*`` are detached by the caller; only the reference carries a graph
  back to the retargeting parameters.  The root pair is weighted by the caller
  through ``weights`` being applied per body: we return the summed loss over all
  correspondences, matching Eq. (17).
  """
  pos_term = weights.pos * torch.sum((reference.pos - simulated_pos) ** 2, dim=-1)
  ori_term = weights.ori * quat_error_magnitude(reference.quat, simulated_quat) ** 2
  loss = pos_term + ori_term
  terms = {"pos": pos_term.sum(dim=-1), "ori": ori_term.sum(dim=-1)}
  if weights.lin_vel > 0.0:
    lin_term = weights.lin_vel * torch.sum(
      (reference.lin_vel - simulated_lin_vel) ** 2, dim=-1
    )
    loss = loss + lin_term
    terms["lin_vel"] = lin_term.sum(dim=-1)
  if weights.ang_vel > 0.0:
    ang_term = weights.ang_vel * torch.sum(
      (reference.ang_vel - simulated_ang_vel) ** 2, dim=-1
    )
    loss = loss + ang_term
    terms["ang_vel"] = ang_term.sum(dim=-1)
  return loss.sum(dim=-1), terms


def deadband(wrench: torch.Tensor, threshold: float) -> torch.Tensor:
  """Eq. (19): continuous deadband, so the policy can output exact zeros."""
  return torch.sign(wrench) * torch.clamp(wrench.abs() - threshold, min=0.0)