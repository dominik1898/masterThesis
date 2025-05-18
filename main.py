import os
import torch
import numpy as np
import random
from typing import cast

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import BaseCallback

import utils
from policies.ensemble_policy import EnsembleActorCriticPolicy
from policies.utils import get_all_ensemble_head_outputs


class UncertaintyMeasurementCallback(BaseCallback):
    """
    Custom callback for measuring aleatoric and epistemic uncertainty during training, collecting real rewards per episode
    and for determining early stopping based on reaching the reward threshold.
    """

    def __init__(self, measurement_interval, total_timesteps, debug=False, verbose=0):
        super(UncertaintyMeasurementCallback, self).__init__(verbose)
        self.debug = debug
        self.terminate_after_timesteps = total_timesteps
        self.terminate_flag = False
        self.uncertainty_measurement_interval = measurement_interval
        self.n_trials = 25
        self.policy_au = []
        self.policy_eu = []
        self.value_eu = []
        self.rewards = []

    def _on_step(self) -> bool:
        """
        Called at every step. Logs uncertainties and rewards, and checks early stopping condition.
        """
        for info in self.locals["infos"]:
            if "episode" in info and "r" in info["episode"]:
                self.rewards.append((self.n_calls, info["episode"]["r"]))

        mean_recent_reward = np.mean([r[1] for r in self.rewards[-self.n_trials:]]) if len(self.rewards) >= self.n_trials else float("-inf")
        if mean_recent_reward >= self.model.get_env().get_attr("reward_threshold")[0] and not self.terminate_flag:
            print("Mean Recent Reward:", mean_recent_reward)
            self.terminate_after_timesteps = int(1.1 * self.n_calls)
            self.terminate_flag = True

        if self.n_calls % self.uncertainty_measurement_interval == 0:
            # Retrieve the current state of the environment in the correct format and shape
            obs = self.locals.get("new_obs")
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.model.device)

            # Compute x y and z for the current measurement interval step and observation
            if isinstance(self.model, (PPO, A2C)):
                policy_eu, value_eu, policy_au = self.model.policy.compute_uncertainties(obs_tensor)
            else:
                policy_eu, value_eu = self.model.policy.compute_uncertainties(obs_tensor)
            if self.debug:
                actions_out, values_out = get_all_ensemble_head_outputs(cast(EnsembleActorCriticPolicy, self.model.policy), obs_tensor)
                if isinstance(actions_out[0], tuple):
                    for i, ((means, std_devs), value) in enumerate(zip(actions_out, values_out)):
                        print(f"Head {i + 1}: Means: {means.cpu().detach().numpy()} | Std-Devs: {std_devs.cpu().detach().numpy()} | Value: {value.cpu().detach().numpy()}")
                else:
                    for i, (actions, value) in enumerate(zip(actions_out, values_out)):
                        print(f"Head {i + 1}: Action Probs: {actions.cpu().detach().numpy()} | Value: {value.cpu().detach().numpy()}")
                print("-----------------------------------------------------------------------------------------------")

            # Save uncertainties to callback lists
            if isinstance(self.model, (PPO, A2C)):
                self.policy_au.append(policy_au)
            self.policy_eu.append(policy_eu)
            self.value_eu.append(value_eu)

        return self.n_calls < self.terminate_after_timesteps


# Ensures reproducibility by setting seeds for Python, NumPy, PyTorch (CPU & GPU), and enforcing deterministic behavior in cuDNN.
base_seed = 42
random.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get the training hyperparameters
args = utils.parse_hyperparameters()

model_dir, results_dir, plots_dir = utils.create_experiment_dirs(args.environment_name, args.agent_name, args.uncertainty_method)

# aggregated uncertainties for all agents
aus_policy, eus_policy, eus_value, all_runs_rewards = [], [], [], []

for agent_idx in range(args.number_of_agents):
    print(f"Training Agent {agent_idx + 1}/{args.number_of_agents}...")

    agent_seed = base_seed + agent_idx

    # Create the environment and initialize the PPO agent
    env, model = utils.initialize_agent(args.agent_name, args.environment_name, agent_seed, args.uncertainty_method, args.ensemble_size, args.debug)

    # Initialize the callback for measuring uncertainties
    callback = UncertaintyMeasurementCallback(measurement_interval=args.uncertainty_measurement_interval, total_timesteps=args.total_timesteps, debug=args.debug)

    # Prints environment, policy, and action distribution details when debug mode is enabled
    if args.debug:
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
        print(f"Policy: {model.policy}")
        print(f"Action Distribution: {model.policy.action_dist}")
        print(f"Action Distribution: {model.policy.get_distribution(torch.tensor(env.reset(), dtype=torch.float32).to(model.device)).distribution}")

        print("Entropy Coefficient:", model.ent_coef)
        print("Max-Grad-Norm:", model.max_grad_norm)
        print("Steps per Update (n_steps):", model.n_steps)
        print("GAE Lambda:", model.gae_lambda)
        print("Value Function Coefficient:", model.vf_coef)
        print("Gamma (discount factor):", model.gamma)
        print("UseRMSProp:", isinstance(model.policy.optimizer, torch.optim.RMSprop))
        print("Normalize Advantage:", model.normalize_advantage)
        print("Learning Rate:", model.learning_rate)
        print("Use SDE:", model.use_sde)

    # Train the model with the callback
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # save the trained model
    model.save(os.path.join(model_dir, f"seed_{agent_idx + 1}"))

    # Save the uncertainties and rewards of the agents after the training
    if isinstance(model, (PPO, A2C)):
        aus_policy.append(callback.policy_au)
    eus_policy.append(callback.policy_eu)
    eus_value.append(callback.value_eu)
    all_runs_rewards.append(callback.rewards)

# save uncertainties and rewards
if args.agent_name in ("PPO", "A2C"):
    np.save(os.path.join(results_dir, "aus_policy.npy"), np.array(aus_policy, dtype=object))
np.save(os.path.join(results_dir, "eus_policy.npy"), np.array(eus_policy, dtype=object))
np.save(os.path.join(results_dir, "eus_value.npy"), np.array(eus_value, dtype=object))
np.save(os.path.join(results_dir, "rewards.npy"), np.array(all_runs_rewards, dtype=object))

utils.plot_rewards(all_runs_rewards, plots_dir)
utils.compute_and_plot_uncertainties(args.uncertainty_measurement_interval, plots_dir, aus_policy, eus_policy, eus_value)
