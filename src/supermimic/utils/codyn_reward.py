"""CoDyn locomotion rewards shared by simulation and imagined rollouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from torch import Tensor, nn

from supermimic.oscillator import JOINT_NAMES
from supermimic.utils.gencoord_helpers import decompose_rep_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


DEFAULT_WEIGHTS = {
  "track_lin_vel_xy": 1.0,
  "track_ang_vel_z": 0.3,
  "hip_knee_tracking": 0.25,
  "flat_orientation": -1.0,
  "base_linear_velocity_z": -0.5,
  "base_angular_velocity_xy": -0.05,
  "joint_velocity": -0.001,
  "action_rate": -0.05,
  "joint_limits": -2.0,
  "hip_yaw_deviation": -0.25,
  "waist_pitch_roll_deviation": -0.5,
  "waist_yaw_deviation": -0.2,
  "arm_deviation": -0.05,
}


class CoDynReward(nn.Module):
  """Score locomotion with soft tracking for the four CoDyn pitch angles.

  Reference order is JOINT_NAMES. Linear commands (vx, vy) use the heading
  frame, and wz is compared with the body-frame yaw rate, as in Isaac Lab.
  Pass the reference and command active for the action being scored.
  Named terms are weighted rates; total rewards include dt.
  """

  def __init__(
    self,
    joint_names: tuple[str, ...],
    nominal_joint_pos: Tensor,
    soft_joint_limits: Tensor,
    dt: float,
    *,
    weights: dict[str, float] | None = None,
    tracking_std: float = 0.4,
  ):
    super().__init__()
    self.dt = dt
    self.weights = DEFAULT_WEIGHTS | (weights or {})
    self.tracking_std = tracking_std
    self.register_buffer("nominal_joint_pos", nominal_joint_pos.detach().clone())
    self.register_buffer("soft_joint_limits", soft_joint_limits.detach().clone())
    groups = {
      "tracking_ids": [joint_names.index(name) for name in JOINT_NAMES],
      "hip_yaw_ids": [
        joint_names.index(f"{side}_hip_yaw_joint") for side in ("left", "right")
      ],
      "waist_tilt_ids": [
        joint_names.index(f"waist_{axis}_joint") for axis in ("pitch", "roll")
      ],
      "waist_yaw_ids": [joint_names.index("waist_yaw_joint")],
      "arm_ids": [
        idx
        for idx, name in enumerate(joint_names)
        if name.split("_")[1] in ("shoulder", "elbow", "wrist")
      ],
    }
    for name, ids in groups.items():
      self.register_buffer(
        name, torch.tensor(ids, dtype=torch.long, device=nominal_joint_pos.device)
      )

  @classmethod
  def from_env(cls, env: ManagerBasedRlEnv, **kwargs) -> CoDynReward:
    robot = env.scene["robot"]
    return cls(
      tuple(robot.joint_names),
      robot.data.default_joint_pos[0],
      robot.data.soft_joint_pos_limits[0],
      env.step_dt if env.cfg.scale_rewards_by_dt else 1.0,
      **kwargs,
    )

  def _terms(
    self,
    joint_pos: Tensor,
    joint_vel: Tensor,
    gravity_b: Tensor,
    root_quat: Tensor,
    lin_vel_b: Tensor,
    ang_vel_b: Tensor,
    reference: Tensor,
    velocity_command: Tensor,
    action: Tensor,
    last_action: Tensor,
  ) -> dict[str, Tensor]:
    lin_vel_yaw = quat_apply_inverse(
      yaw_quat(root_quat), quat_apply(root_quat, lin_vel_b)
    )
    lin_error = (lin_vel_yaw[..., :2] - velocity_command[..., :2]).square().sum(-1)
    yaw_error = (ang_vel_b[..., 2] - velocity_command[..., 2]).square()
    upright = (-gravity_b[..., 2]).clamp(0.0, 0.7) / 0.7
    hip_knee_error = (joint_pos[..., self.tracking_ids] - reference).square()
    deviation = (joint_pos - self.nominal_joint_pos).abs()
    lo, hi = self.soft_joint_limits.unbind(-1)
    terms = {
      "track_lin_vel_xy": (-lin_error / 0.25).exp() * upright,
      "track_ang_vel_z": (-yaw_error / 0.25).exp(),
      "hip_knee_tracking": (-hip_knee_error / self.tracking_std**2).exp().mean(-1),
      "flat_orientation": gravity_b[..., :2].square().sum(-1),
      "base_linear_velocity_z": lin_vel_b[..., 2].square(),
      "base_angular_velocity_xy": ang_vel_b[..., :2].square().sum(-1),
      "joint_velocity": joint_vel.square().sum(-1),
      "action_rate": (action - last_action).square().sum(-1),
      "joint_limits": (
        (lo - joint_pos).clamp_min(0) + (joint_pos - hi).clamp_min(0)
      ).sum(-1),
      "hip_yaw_deviation": deviation[..., self.hip_yaw_ids].sum(-1),
      "waist_pitch_roll_deviation": deviation[..., self.waist_tilt_ids].sum(-1),
      "waist_yaw_deviation": deviation[..., self.waist_yaw_ids].sum(-1),
      "arm_deviation": deviation[..., self.arm_ids].sum(-1),
    }
    return terms

  def _weighted(self, terms: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value * self.weights[name] for name, value in terms.items()}

  def sim_terms(
    self,
    env: ManagerBasedRlEnv,
    reference: Tensor,
    velocity_command: Tensor,
    *,
    weighted: bool = True,
  ) -> dict[str, Tensor]:
    data = env.scene["robot"].data
    terms = self._terms(
      data.joint_pos,
      data.joint_vel,
      data.projected_gravity_b,
      data.root_link_quat_w,
      data.root_link_lin_vel_b,
      data.root_link_ang_vel_b,
      reference,
      velocity_command,
      env.action_manager.action,
      env.action_manager.prev_action,
    )
    return self._weighted(terms) if weighted else terms

  def imagined_terms(
    self,
    state: Tensor,
    root_quat: Tensor,
    reference: Tensor,
    velocity_command: Tensor,
    action: Tensor,
    last_action: Tensor,
  ) -> dict[str, Tensor]:
    _, gravity_b, lin_vel_b, ang_vel_b, joint_pos, joint_vel = decompose_rep_state(
      state
    )
    terms = self._terms(
      joint_pos,
      joint_vel,
      gravity_b,
      root_quat,
      lin_vel_b,
      ang_vel_b,
      reference,
      velocity_command,
      action,
      last_action,
    )
    return self._weighted(terms)

  def forward(
    self,
    state: Tensor,
    root_quat: Tensor,
    reference: Tensor,
    velocity_command: Tensor,
    action: Tensor,
    last_action: Tensor,
  ) -> Tensor:
    terms = self.imagined_terms(
      state, root_quat, reference, velocity_command, action, last_action
    )
    return self.dt * sum(terms.values())
