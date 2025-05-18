import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from policies.ensemble_policy import EnsembleActorCriticPolicy


class RewardThresholdCallback(BaseCallback):
    def __init__(self, threshold, window_size):
        super().__init__()
        self.threshold = threshold
        self.window_size = window_size
        self.recent_rewards = []
        self.episode_rewards = []
        self.episode_timesteps = []

    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    ep_reward = info["episode"]["r"]
                    self.recent_rewards.append(ep_reward)
                    if len(self.recent_rewards) > self.window_size:
                        self.recent_rewards.pop(0)
                    avg_reward = np.mean(self.recent_rewards)
                    self.episode_rewards.append(ep_reward)
                    self.episode_timesteps.append(self.num_timesteps)
                    if len(self.recent_rewards) == self.window_size and avg_reward >= self.threshold:
                        return False
        return True


env = gym.make("Acrobot-v1")
ensemble_sizes = [1, 4, 7]
callbacks = {}

for ensemble_size in ensemble_sizes:
    model = PPO(EnsembleActorCriticPolicy, env, verbose=1, policy_kwargs={"ensemble_size": ensemble_size})
    callback = RewardThresholdCallback(threshold=-100.0, window_size=25)
    model.learn(total_timesteps=1_000_000, callback=callback)
    callbacks[ensemble_size] = callback

env.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, ensemble_size in zip(axes, ensemble_sizes):
    callback = callbacks[ensemble_size]
    ax.plot(callback.episode_timesteps, callback.episode_rewards, linestyle="-", linewidth=1.5)
    ax.set_xlabel("Total Timesteps")
    if ensemble_size == 1:
        ax.set_ylabel("Reward")
    ax.set_title(f"Ensemble Size: {ensemble_size}", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

plt.tight_layout()
plt.savefig("acrobot_ensemble_comparison.pdf", dpi=600)
plt.show()
