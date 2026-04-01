from __future__ import annotations

import os
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
    sync_envs_normalization,
)


def _evaluate_policy_with_tqdm(
    model,
    env: Union[gym.Env, VecEnv],
    *,
    n_eval_episodes: int,
    deterministic: bool,
    render: bool,
    callback,
    warn: bool,
    progress_desc: str,
):
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    if not is_monitor_wrapped and warn:
        import warnings

        warnings.warn(
            "Evaluation environment is not wrapped with a `Monitor` wrapper. "
            "Reported episode lengths/rewards may be wrapper-modified.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_infos = []
    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    bar = None
    if tqdm is not None:
        bar = tqdm(total=int(n_eval_episodes), desc=progress_desc, leave=False, dynamic_ncols=True)

    try:
        while (episode_counts < episode_count_targets).any():
            actions, states = model.predict(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            new_observations, rewards, dones, infos = env.step(actions)
            current_rewards += rewards
            current_lengths += 1

            for i in range(n_envs):
                if episode_counts[i] < episode_count_targets[i]:
                    done = dones[i]
                    info = infos[i]
                    episode_starts[i] = done
                    if callback is not None:
                        callback({"done": done, "info": info}, {})

                    if done:
                        if is_monitor_wrapped:
                            if "episode" in info:
                                episode_rewards.append(float(info["episode"]["r"]))
                                episode_lengths.append(int(info["episode"]["l"]))
                                episode_infos.append(dict(info))
                                episode_counts[i] += 1
                                if bar is not None:
                                    bar.update(1)
                        else:
                            episode_rewards.append(float(current_rewards[i]))
                            episode_lengths.append(int(current_lengths[i]))
                            episode_infos.append(dict(info))
                            episode_counts[i] += 1
                            if bar is not None:
                                bar.update(1)
                        current_rewards[i] = 0.0
                        current_lengths[i] = 0

            observations = new_observations
            if render:
                env.render()
    finally:
        if bar is not None:
            bar.close()

    return episode_rewards, episode_lengths, episode_infos


class TqdmEvalCallback(EvalCallback):
    """EvalCallback variant that shows per-episode eval progress with tqdm."""

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way."
                    ) from e

            self._is_success_buffer = []
            episode_rewards, episode_lengths, episode_infos = _evaluate_policy_with_tqdm(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                warn=self.warn,
                callback=self._log_success_callback,
                progress_desc=f"Eval @ step {self.num_timesteps}",
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                if not hasattr(self, "evaluations_true_cycle_area_norm"):
                    self.evaluations_true_cycle_area_norm = []
                if not hasattr(self, "evaluations_largest_true_cycle_area_norm"):
                    self.evaluations_largest_true_cycle_area_norm = []
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                final_true_cycle_area_norm = [float(info.get("true_cycle_area_norm", 0.0)) for info in episode_infos]
                final_largest_true_cycle_area_norm = [
                    float(info.get("largest_true_cycle_area_norm", 0.0)) for info in episode_infos
                ]
                self.evaluations_true_cycle_area_norm.append(final_true_cycle_area_norm)
                self.evaluations_largest_true_cycle_area_norm.append(final_largest_true_cycle_area_norm)
                kwargs: dict[str, Any] = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    true_cycle_area_norm=self.evaluations_true_cycle_area_norm,
                    largest_true_cycle_area_norm=self.evaluations_largest_true_cycle_area_norm,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
