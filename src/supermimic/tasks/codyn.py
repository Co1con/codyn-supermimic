"""CoDyn walking from the default G1 pose, with online hip–knee targets."""

import mjlab.envs.mdp as mdp
import torch
from mjlab.managers import (
  EventTermCfg,
  ManagerTermBase,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)

from supermimic.oscillator import CoDynGaitCommandCfg
from supermimic.tasks.env_cfgs import gencoord_env_cfg
from supermimic.utils.codyn_reward import DEFAULT_WEIGHTS, CoDynReward

TASK_ID = "CoDyn-GenCoord-G1"
MIN_HEIGHT = 0.45
MAX_TILT = 0.8


def codyn_goal(env):
  gait = env.command_manager.get_term("codyn")
  velocity = torch.zeros_like(gait.command[:, :3])
  velocity[:, 0] = gait.cfg.forward_speed
  return torch.cat((gait.command, velocity), dim=-1)


class _RewardTerm(ManagerTermBase):
  """Cache the common calculation across the thirteen manager terms in a step."""

  def __init__(self, cfg, env):
    super().__init__(env)
    if not hasattr(env, "_codyn_rewards"):
      env._codyn_rewards = (CoDynReward.from_env(env), -1, {})

  def __call__(self, env, name):
    reward, step, values = env._codyn_rewards
    if step != env.common_step_counter:
      goal = codyn_goal(env)
      values = reward.sim_terms(env, goal[:, :4], goal[:, 4:], weighted=False)
      env._codyn_rewards = (reward, env.common_step_counter, values)
    return values[name]


def codyn_env_cfg(play: bool = False):
  cfg = gencoord_env_cfg(play=play)
  cfg.commands = {"codyn": CoDynGaitCommandCfg(checkpoint="")}
  cfg.observations.pop("critic")
  cfg.observations["goal"].terms = {"codyn": ObservationTermCfg(func=codyn_goal)}
  cfg.events["reset_default"] = EventTermCfg(
    func=mdp.reset_scene_to_default, mode="reset"
  )
  cfg.rewards = {
    name: RewardTermCfg(func=_RewardTerm, params={"name": name}, weight=weight)
    for name, weight in DEFAULT_WEIGHTS.items()
  }
  cfg.terminations = {
    "time_out": cfg.terminations["time_out"],
    "low_base": TerminationTermCfg(
      func=mdp.root_height_below_minimum, params={"minimum_height": MIN_HEIGHT}
    ),
    "tilt": TerminationTermCfg(
      func=mdp.bad_orientation, params={"limit_angle": MAX_TILT}
    ),
  }
  return cfg
