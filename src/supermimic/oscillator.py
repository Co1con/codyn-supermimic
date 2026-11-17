"""CoDyn hip–knee oscillator with batched G1 references for mjlab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from torch import Tensor, nn

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_knee_joint",
  "right_hip_pitch_joint",
  "right_knee_joint",
)


class _SpeedOscillator(nn.Module):
  """The trained two-dimensional vector field for a single leg."""

  def __init__(self, geometry: dict, bounded_compensation: bool):
    super().__init__()
    self.register_buffer(
      "geometry_base", torch.tensor(geometry["coefficients"], dtype=torch.float32)
    )
    self.register_buffer(
      "geometry_scale", torch.tensor([0.10, 0.10, 0.25, 0.25, 0.25, 0.25])
    )
    self.midpoint = geometry["midpoint"]
    self.half_range = geometry["half_range"]
    self.sign = geometry["sign"]
    self.bounded_compensation = bounded_compensation
    hidden = 48
    self.speed_net = nn.Sequential(
      nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 6 + 2 * hidden)
    )
    self.state_net = nn.Sequential(
      nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
    )
    self.shaping = nn.Linear(hidden, 5)

  def parameters_at(self, speed: Tensor):
    normalized = (speed.reshape(-1, 1) - self.midpoint) / self.half_range
    condition = self.speed_net(normalized)
    base = (
      torch.cat([torch.ones_like(normalized), normalized], dim=1) @ self.geometry_base
    )
    parameters = base + self.geometry_scale * condition[:, :6].tanh()
    center, axes = parameters[:, :2], parameters[:, 2:4].exp()
    theta, frequency = parameters[:, 4], parameters[:, 5].exp()
    cos, sin = theta.cos(), theta.sin()
    basis = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], 1)
    return center, axes, basis, frequency, condition[:, 6:].chunk(2, dim=1)

  def field(self, q: Tensor, parameters) -> Tensor:
    center, axes, basis, frequency, (gain, bias) = parameters
    offset = q - center
    rotated = torch.bmm(offset[:, None], basis).squeeze(1)
    normalized = rotated / axes
    gradient = torch.bmm(
      (rotated / axes.square())[:, None], basis.transpose(1, 2)
    ).squeeze(1)
    hidden = self.state_net(normalized)
    output = self.shaping(hidden * (1 + 0.25 * gain.tanh()) + 0.25 * bias.tanh())
    a, b, c = (0.1 * output[:, :3].tanh()).unbind(-1)
    lower = torch.stack(
      [torch.stack([a, torch.zeros_like(a)], -1), torch.stack([b, c], -1)], 1
    )
    damping = torch.bmm(lower @ lower.transpose(1, 2), gradient[:, :, None]).squeeze(-1)
    tangent = torch.stack([-gradient[:, 1], gradient[:, 0]], -1)
    circulation = (
      self.sign
      * 2
      * torch.pi
      * frequency[:, None]
      * axes.prod(-1, keepdim=True)
      * tangent
    )
    radial = 5 * (normalized.square().sum(-1, keepdim=True) - 1) * offset
    compensation = output[:, 3:].tanh() if self.bounded_compensation else output[:, 3:]
    return circulation - radial - damping + 2 * compensation


class CoDynOscillator(nn.Module):
  """Batched angles and rates in G1 coordinates, ordered by JOINT_NAMES.

  The caller holds each environment's oscillator state. Joint tensors have shape
  (num_envs, 4), speed has shape (num_envs,) in m/s, and dt is in seconds.
  """

  def __init__(
    self, checkpoint: str | Path, substeps: int = 4, coupling_gain: float = 0.25
  ):
    super().__init__()
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    self.model = _SpeedOscillator(data["geometry"], data["version"] == 1)
    self.model.load_state_dict(data["model_state"])
    self.model.requires_grad_(False)
    self.register_buffer("phase_bank", data["phase_bank"][:, :-1])
    self.register_buffer("phase_speeds", data["phase_speeds"])
    self.register_buffer("signs", torch.tensor([-1.0, 1.0, -1.0, 1.0]))
    self.substeps = substeps
    self.coupling_gain = coupling_gain

  def _bank_at(self, speed: Tensor) -> Tensor:
    speed = speed.clamp(self.phase_speeds[0], self.phase_speeds[-1])
    high = torch.searchsorted(self.phase_speeds, speed.contiguous()).clamp(
      1, len(self.phase_speeds) - 1
    )
    low = high - 1
    fraction = (speed - self.phase_speeds[low]) / (
      self.phase_speeds[high] - self.phase_speeds[low]
    )
    return torch.lerp(
      self.phase_bank[low], self.phase_bank[high], fraction[:, None, None]
    )

  def initial_state(self, speed: Tensor, phase_index: Tensor | None = None) -> Tensor:
    """Return angles in radians, starting the right leg half a cycle ahead."""
    bank = self._bank_at(speed)
    count = bank.shape[1]
    if phase_index is None:
      phase_index = torch.zeros_like(speed, dtype=torch.long)
    indices = torch.stack([phase_index, phase_index + count // 2], dim=1) % count
    batch = torch.arange(len(speed), device=speed.device)[:, None]
    return bank[batch, indices].flatten(1) * self.signs

  def _coupling_rates(self, state: Tensor, speed: Tensor) -> Tensor:
    if self.coupling_gain == 0:
      return torch.ones_like(state[..., :1]).reshape(-1, 1)
    bank = self._bank_at(speed)
    scale = bank.std(dim=1, unbiased=False).clamp_min(1e-6)
    points = bank / scale[:, None]
    segments = points.roll(-1, dims=1) - points
    offset = state[:, :, None] / scale[:, None, None] - points[:, None]
    fraction = (offset * segments[:, None]).sum(-1) / segments.square().sum(-1)[
      :, None
    ].clamp_min(1e-12)
    fraction = fraction.clamp(0, 1)
    distance = (offset - fraction[..., None] * segments[:, None]).square().sum(-1)
    index = distance.argmin(dim=-1)
    fraction = fraction.gather(-1, index[..., None]).squeeze(-1)
    phases = (index + fraction) * (2 * torch.pi / bank.shape[1])
    correction = self.coupling_gain * (phases[:, 1] - phases[:, 0] - torch.pi).sin()
    return torch.stack([1 + correction, 1 - correction], dim=1).reshape(-1, 1)

  def forward(self, q: Tensor, speed: Tensor) -> Tensor:
    """Evaluate the reference angular velocity in rad/s."""
    state = (q * self.signs).reshape(-1, 2, 2)
    parameters = self.model.parameters_at(speed.repeat_interleave(2))
    velocity = self.model.field(state.reshape(-1, 2), parameters)
    return (velocity * self._coupling_rates(state, speed)).reshape_as(q) * self.signs

  def step(self, q: Tensor, speed: Tensor, dt: float | Tensor) -> Tensor:
    """Advance the reference angles with RK4 for one control interval."""
    state = (q * self.signs).reshape(-1, 2, 2)
    parameters = self.model.parameters_at(speed.repeat_interleave(2))
    rates = self._coupling_rates(state, speed)
    h = dt / self.substeps
    if isinstance(h, Tensor):
      h = h.repeat_interleave(2).unsqueeze(1)
    state = state.reshape(-1, 2)

    def field(value: Tensor) -> Tensor:
      return self.model.field(value, parameters) * rates

    for _ in range(self.substeps):
      k1 = field(state)
      k2 = field(state + h * k1 / 2)
      k3 = field(state + h * k2 / 2)
      k4 = field(state + h * k3)
      state = state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return state.reshape_as(q) * self.signs


@dataclass(kw_only=True)
class CoDynGaitCommandCfg(CommandTermCfg):
  checkpoint: str | Path
  forward_speed: float = 1.0
  speed_command_name: str | None = None
  """Optional velocity command key. Its first column is vx; register it before this term."""
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  def build(self, env: ManagerBasedRlEnv) -> CoDynGaitCommand:
    return CoDynGaitCommand(self, env)


class CoDynGaitCommand(CommandTerm):
  """Advance one independent reference per environment through CommandManager."""

  cfg: CoDynGaitCommandCfg

  def __init__(self, cfg: CoDynGaitCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.oscillator = CoDynOscillator(cfg.checkpoint).to(self.device).eval()
    self._fixed_speed = torch.full(
      (self.num_envs,), cfg.forward_speed, device=self.device
    )
    self._position = self.oscillator.initial_state(self._fixed_speed)
    self.joint_vel = self.oscillator(self._position, self._fixed_speed)
    self._dt: float | Tensor = 0.0

  @property
  def command(self) -> Tensor:
    """Four position targets in radians, ordered by JOINT_NAMES."""
    return self._position

  def _speed(self, ids: slice | Tensor) -> Tensor:
    if self.cfg.speed_command_name is None:
      return self._fixed_speed[ids]
    return self._env.command_manager.get_command(self.cfg.speed_command_name)[ids, 0]

  @torch.no_grad()
  def compute(self, dt: float | Tensor, env_ids: Tensor | None = None) -> None:
    # The manager supplies zero dt for just-reset environments.
    self._dt = dt
    super().compute(dt, env_ids)

  def _update_metrics(self) -> None:
    pass

  @torch.no_grad()
  def _resample_command(self, env_ids: Tensor) -> None:
    speed = self._speed(env_ids)
    position = self.oscillator.initial_state(speed)
    self._position[env_ids] = position
    self.joint_vel[env_ids] = self.oscillator(position, speed)

  def _update_command(self, env_ids: Tensor | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    speed = self._speed(ids)
    position = self.oscillator.step(self._position[ids], speed, self._dt)
    self._position[ids] = position
    self.joint_vel[ids] = self.oscillator(position, speed)
