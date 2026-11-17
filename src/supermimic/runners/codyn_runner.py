"""SuperMimic training for the online CoDyn gait task."""

from math import cos

import torch

from supermimic.envs.world_model_env import WorldModelEnv
from supermimic.replay_buffer import FieldSpec
from supermimic.runners.supermimic_runner import SuperMimicRunner, build_actor
from supermimic.tasks.task_spec import TaskSpec, build_actor_obs_td
from supermimic.utils.codyn_reward import CoDynReward


class CoDynWorldModelEnv(WorldModelEnv):
  def __init__(self, *args, task_specs, **kwargs):
    self.goal_name = task_specs[0].name
    super().__init__(*args, **kwargs)

  def _bind_task(self, env, termination_softness):
    self._num_actions = env.num_actions
    real_env = env.unwrapped
    self.step_dt = real_env.step_dt
    self.oscillator = real_env.command_manager.get_term("codyn").oscillator
    self.reward = CoDynReward.from_env(
      real_env, weights={name: term.weight for name, term in env.cfg.rewards.items()}
    )
    self.min_height = env.cfg.terminations["low_base"].params["minimum_height"]
    self.min_upright = cos(env.cfg.terminations["tilt"].params["limit_angle"])
    self.termination_softness = termination_softness

  def _on_reset(self, batch):
    self._wm_goal_obs = batch["codyn_goal"]
    if self.history_length > 1:
      self._past_sensors = self._warmstart_past_sensors(
        batch["sensor_window"], batch["pad_mask"]
      )
    self._latest_sensor = self._synth_sensor()

  @torch.no_grad()
  def _compute_commands(self):
    reference, command = self._wm_goal_obs.split((4, 3), dim=-1)
    next_reference = self.oscillator.step(reference, command[:, 0], self.step_dt)
    self._wm_goal_obs = torch.cat((next_reference, command), dim=-1)

  def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
    next_aux, next_state = self.dynamics(rep_state, action, aux_state, sim_ctx)
    reference, command = self._wm_goal_obs.split((4, 3), dim=-1)
    reward = self.reward(
      next_state, next_aux[:, 3:7], reference, command, action, last_action
    )
    height_error = (self.min_height - next_aux[:, 2]) / self.min_height
    tilt_error = (self.min_upright + next_state[:, 3]) / (1.0 - self.min_upright)
    term_prob = torch.sigmoid(
      torch.maximum(height_error, tilt_error) / self.termination_softness
    )
    return (
      next_aux,
      next_state,
      self.representation.joint_state(next_state),
      term_prob,
      reward,
    )

  def get_observations(self):
    return build_actor_obs_td(
      {self.goal_name: self._wm_goal_obs},
      self._sensor_history(),
      critic=self._critic_obs(),
    )


class CoDynRunner(SuperMimicRunner):
  world_model_env_cls = CoDynWorldModelEnv

  def _make_policy_actor(self):
    return build_actor(
      self.cfg, self._wm, self.device, task_specs=[TaskSpec("codyn", self._num_goals)]
    )

  def _configure_reset_sampling(self):
    # The task resets to the default G1 pose and oscillator initial state.
    pass

  def _buffer_schema(self):
    schema = super()._buffer_schema()
    del schema["motion_time"]
    schema["codyn_goal"] = FieldSpec(self._num_goals)
    return schema

  def step(self):
    with torch.inference_mode():
      obs = self.env.get_observations()
      full_state, goal = obs["full_state"], obs["goal"]
      representation = self._representation.represent(full_state)
      sim_ctx = self._sim_context.read(self.env.unwrapped)
      sensor = self._representation.compute_actor_sensor(
        representation,
        sim_ctx,
        sim_actor_obs=obs["actor"],
        noisy=self._stores_actor_obs,
      )
      actor_td = build_actor_obs_td({"codyn": goal}, self._collection_history(sensor))
      action = self.policy_learner.actor(actor_td)
      action = action + torch.randn_like(action) * self.policy_learner.cfg.noise
      fields = {"sim_ctx": sim_ctx, "codyn_goal": goal.clone()}
      if self._stores_actor_obs:
        fields["actor_obs"] = obs["actor"]
      next_obs, rewards, dones, extras = self._step_and_store(
        action, full_state, actor_td=actor_td, extra_fields=fields
      )
      self._advance_sim_history(next_obs["actor"], dones)
      self.logger.process_env_step(rewards, dones.bool(), extras)
    self._flush_normalization()
    return self.env.num_envs
