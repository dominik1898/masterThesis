import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn # important for eval(policy_kwargs) in initialize_agent methods
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, TransformObservation, ResizeObservation, GrayscaleObservation
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.atari_wrappers import NoopResetEnv, EpisodicLifeEnv, FireResetEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import stable_baselines3.common.utils as sb3_utils
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike # important for eval(policy_kwargs) in initialize_a2c_agent method
import argparse
import inspect
import yaml
import ale_py
gym.register_envs(ale_py)
from policies.ensemble_policy import EnsembleActorCriticPolicy
from policies.dropout_policy import DropoutActorCriticPolicy
from policies.ddpg_policies import DropoutDDPGPolicy, EnsembleDDPGPolicy, EnsembleDDPG

def parse_hyperparameters():
    """
    Parses hyperparameters from the command line input.

    :return: Parsed arguments containing all hyperparameters
    """
    parser = argparse.ArgumentParser(description="Parse hyperparameters for RL training with various agents and uncertainty estimation techniques")

    # Number of agents to train (e.g., for different seeds or ensemble evaluations)
    parser.add_argument('--number_of_agents', type=int, default=3, help='Number of agents to train')
    # Total number of timesteps for training each agent (8192 16384   32768   65536   131072  262144  524288  1048576   2097152 4194304 8388608 16777216    33554432    67108864)
    parser.add_argument('--total_timesteps', type=int, default=16_777_216, help='Total number of training timesteps')
    # Interval for measuring the different uncertainties during training (in timesteps)
    parser.add_argument('--uncertainty_measurement_interval', type=int, default=2048, help='Interval for measuring uncertainty (in timesteps)')
    # Environment name from Gymnasium
    parser.add_argument('--environment_name', type=str, required=True, help='Name of the environment to train on')
    # Agent name from stable-baselines3
    parser.add_argument('--agent_name', type=str, required=True, help='Name of the agent actor-critic algorithm')
    # Uncertainty method (e.g., Ensemble, MC-Dropout, etc.)
    parser.add_argument('--uncertainty_method', type=str, required=True, help='Uncertainty estimation method')
    # If --uncertainty_method equals "Ensemble": size of the ensemble (number of policy- and value-heads in the ensemble actor-critic network)
    parser.add_argument('--ensemble_size', type=int, default=7, help='Number of heads in the ensemble policy (--uncertainty_method=="Ensemble")')
    # Debug mode to print critical information for inspecting the environment, policy, action distribution, and uncertainty values
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to print critical information during training')
    args = parser.parse_args()
    return args


def create_experiment_dirs(environment_name: str, agent_name: str, uncertainty_method: str):
    """
    Creates directories for saving models, results, and plots.

    :param environment_name: Name of the environment (e.g., "CartPole").
    :param agent_name: Name of the agent (e.g., "PPO").
    :param uncertainty_method: The method used for uncertainty estimation (e.g., "Ensemble").
    :return: Tuple containing paths to (model_dir, results_dir, plots_dir).
    """
    base_dir = os.getcwd()
    model_dir = os.path.join(base_dir, "models", environment_name, agent_name, uncertainty_method)
    results_dir = os.path.join(base_dir, "results", environment_name, agent_name, uncertainty_method)
    plots_dir = os.path.join(base_dir, "plots", environment_name, agent_name, uncertainty_method)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return model_dir, results_dir, plots_dir


def create_environment(environment_name: str, seed: int = None) -> VecEnv:
    """
    Creates and returns the appropriate environment based on the provided environment name.
    Also assigns an environment-specific reward threshold for early stopping.

    :param environment_name: Name of the environment as a string
    :param seed: Random seed for environment reproducibility
    :return: The corresponding vectorized Gymnasium environment (DummyVecEnv)
    """
    if environment_name == "Acrobot":
        env = gym.make("Acrobot-v1")
    elif environment_name == "CartPole":
        env = gym.make("CartPole-v1")
    elif environment_name == "MountainCarContinuous":
        env = gym.make("MountainCarContinuous-v0")
    elif environment_name == "MountainCar":
        env = gym.make("MountainCar-v0")
    elif environment_name == "Pendulum":
        env = gym.make("Pendulum-v1")
    elif environment_name == "BipedalWalker":
        env = gym.make("BipedalWalker-v3")
    elif environment_name == "CarRacing":
        env = gym.make("CarRacing-v3")
        env = ResizeObservation(env, shape=(64, 64))
        env = GrayscaleObservation(env)
        env = FrameStackObservation(env, 2)
    elif environment_name == "LunarLander":
        env = gym.make("LunarLander-v3", continuous=True)
    elif environment_name == "Blackjack":
        env = gym.make("Blackjack-v1")
        env = TransformObservation(env, lambda obs: np.array(obs, dtype=np.int32), observation_space=gym.spaces.MultiDiscrete([32, 11, 2])) # Converts tuple observations into a MultiDiscrete numpy array for compatibility.
    elif environment_name == "Taxi":
        env = gym.make('Taxi-v3')
    elif environment_name == "CliffWalking":
        env = gym.make('CliffWalking-v0')
    elif environment_name == "FrozenLake":
        env = gym.make('FrozenLake-v1', is_slippery=True)
    elif environment_name == "Ant":
        env = gym.make("Ant-v5")
    elif environment_name == "HalfCheetah":
        env = gym.make("HalfCheetah-v5")
    elif environment_name == "Hopper":
        env = gym.make("Hopper-v5")
    elif environment_name == "Humanoid":
        env = gym.make("Humanoid-v5")
    elif environment_name == "HumanoidStandup":
        env = gym.make("HumanoidStandup-v5")
    elif environment_name == "InvertedDoublePendulum":
        env = gym.make("InvertedDoublePendulum-v5")
    elif environment_name == "InvertedPendulum":
        env = gym.make("InvertedPendulum-v5")
    elif environment_name == "Pusher":
        env = gym.make("Pusher-v5")
    elif environment_name == "Reacher":
        env = gym.make("Reacher-v5")
    elif environment_name == "Swimmer":
        env = gym.make("Swimmer-v5")
    elif environment_name == "Walker2d":
        env = gym.make("Walker2d-v5")
    elif environment_name.startswith("ALE/"):
        env = gym.make(environment_name + "-v5")
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env)
        env = FrameStackObservation(env, 4)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        if hasattr(env.unwrapped, "get_action_meanings"):
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
    else:
        raise ValueError(f"Unsupported environment: {environment_name}")
    reward_threshold = env.spec.reward_threshold
    if reward_threshold is None:
        manual_thresholds = {"Pendulum": -130.0, "Blackjack": 0.7, "CliffWalking": -15.0, "Humanoid": 12600.0, "HumanoidStandup": 320000.0, "Walker2d": 5000.0}
        reward_threshold = manual_thresholds.get(environment_name, float("inf"))
    env.reward_threshold = reward_threshold
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return DummyVecEnv([lambda: env])


def initialize_ppo_agent(environment_name: str, agent_seed: int, uncertainty_method: str, ensemble_size: int, debug: bool) -> tuple:
    """
    Initializes a PPO agent using hyperparameters from a YAML configuration file.
    This method loads default PPO parameters, overrides them with environment-specific settings from a YAML file,
    sets up the environment (with optional normalization and feature extraction settings), and initializes the PPO model.

    :param environment_name: Name of the Gymnasium environment to train on.
    :param agent_seed: Random seed for reproducibility.
    :param uncertainty_method: The uncertainty estimation method to use.
    :param ensemble_size: The size of the ensemble (if applicable).
    :param debug: Whether to enable verbose output.
    :return: A tuple (environment, PPO model) where the environment is the configured Gymnasium environment
             and the PPO model is the initialized RL agent.
    """
    # Load Default PPO Parameters
    ppo_defaults = {k: v.default for k, v in inspect.signature(PPO.__init__).parameters.items() if v.default is not inspect.Parameter.empty}

    # Load environment-specific YAML Hyperparameters
    with open("hyperparams/ppo.yaml", "r") as file:
        hyperparams = yaml.safe_load(file)
    env_hyperparams = hyperparams.get(environment_name, {})
    if environment_name.startswith("ALE/"):
        env_hyperparams = hyperparams.get("ALE/", env_hyperparams)

    # Merge the PPO default parameters with the environment-specific YAML parameters.
    ppo_params = {**ppo_defaults, **env_hyperparams}

    # Convert linear learning rate decay notation from YAML if applicable.
    if isinstance(ppo_params["learning_rate"], str):
        if ppo_params["learning_rate"].startswith("lin_"):
            lr_value = float(ppo_params["learning_rate"].replace("lin_", ""))
            ppo_params["learning_rate"] = sb3_utils.get_linear_fn(lr_value, 0, 1)
    # Check if clip_range is a string and follows the "lin_xx" format
    if isinstance(ppo_params["clip_range"], str):
        if ppo_params["clip_range"].startswith("lin_"):
            clip_value = float(ppo_params["clip_range"].replace("lin_", ""))
            ppo_params["clip_range"] = sb3_utils.get_linear_fn(clip_value, 0, 1)

    # Process Policy Keyword Arguments (policy_kwargs)
    yaml_policy_kwargs = env_hyperparams.get("policy_kwargs", "{}")
    if isinstance(yaml_policy_kwargs, str):
        yaml_policy_kwargs = eval(yaml_policy_kwargs)
    if uncertainty_method == "Ensemble":
        yaml_policy_kwargs["ensemble_size"] = ensemble_size
    if ppo_params.get("policy") == "CnnPolicy":
        yaml_policy_kwargs["features_extractor_class"] = NatureCNN

    # Create Environment
    env = create_environment(environment_name, agent_seed)
    if ppo_params.get("normalize") is True:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    elif ppo_params.get("normalize")=="{'norm_obs': False, 'norm_reward': True}":
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Remove Unnecessary Parameters From PPO-Parameters
    for key in ["policy", "normalize", "policy_kwargs", "verbose", "seed"]:
        ppo_params.pop(key, None)

    # Initialize PPO Model
    model = None
    if uncertainty_method == "Ensemble":
        model = PPO(EnsembleActorCriticPolicy, env, verbose=0 if debug else 1, policy_kwargs=yaml_policy_kwargs, seed=agent_seed, **ppo_params)
    if uncertainty_method == "Dropout":
        model = PPO(DropoutActorCriticPolicy, env, verbose=0 if debug else 1, policy_kwargs=yaml_policy_kwargs, seed=agent_seed, **ppo_params)

    return env, model


def initialize_a2c_agent(environment_name: str, agent_seed: int, uncertainty_method: str, ensemble_size: int, debug: bool) -> tuple:
    """
    Initializes an A2C agent using hyperparameters from a YAML configuration file.
    This method loads default A2C parameters, overrides them with environment-specific settings from a YAML file,
    sets up the environment (with optional normalization and feature extraction settings), and initializes the A2C model.

    :param environment_name: Name of the Gymnasium environment to train on.
    :param agent_seed: Random seed for reproducibility.
    :param uncertainty_method: The uncertainty estimation method to use.
    :param ensemble_size: The size of the ensemble (if applicable).
    :param debug: Whether to enable verbose output.
    :return: A tuple (environment, A2C model) where the environment is the configured Gymnasium environment
             and the A2C model is the initialized RL agent.
    """
    # Load Default A2C Parameters
    a2c_defaults = {k: v.default for k, v in inspect.signature(A2C.__init__).parameters.items() if v.default is not inspect.Parameter.empty}

    # Load environment-specific YAML Hyperparameters
    with open("hyperparams/a2c.yaml", "r") as file:
        hyperparams = yaml.safe_load(file)
    env_hyperparams = hyperparams.get(environment_name, {})
    if environment_name.startswith("ALE/"):
        env_hyperparams = hyperparams.get("ALE/", env_hyperparams)

    # Merge the A2C default parameters with the environment-specific YAML parameters.
    a2c_params = {**a2c_defaults, **env_hyperparams}

    # Convert linear learning rate decay notation from YAML if applicable.
    if isinstance(a2c_params["learning_rate"], str):
        if a2c_params["learning_rate"].startswith("lin_"):
            lr_value = float(a2c_params["learning_rate"].replace("lin_", ""))
            a2c_params["learning_rate"] = sb3_utils.get_linear_fn(lr_value, 0, 1)

    # Process Policy Keyword Arguments (policy_kwargs)
    yaml_policy_kwargs = env_hyperparams.get("policy_kwargs", "{}")
    if isinstance(yaml_policy_kwargs, str):
        yaml_policy_kwargs = eval(yaml_policy_kwargs)
    if uncertainty_method == "Ensemble":
        yaml_policy_kwargs["ensemble_size"] = ensemble_size
    if a2c_params.get("policy") == "CnnPolicy":
        yaml_policy_kwargs["features_extractor_class"] = NatureCNN

    # Create Environment
    env = create_environment(environment_name, agent_seed)
    if a2c_params.get("normalize") is True:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    elif a2c_params.get("normalize")=="{'norm_obs': False, 'norm_reward': True}":
        env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # Remove Unnecessary Parameters From A2C-Parameters
    for key in ["policy", "normalize", "policy_kwargs", "verbose", "seed"]:
        a2c_params.pop(key, None)

    # Initialize A2C Model
    model = None
    if uncertainty_method == "Ensemble":
        model = A2C(EnsembleActorCriticPolicy, env, verbose=0 if debug else 1, policy_kwargs=yaml_policy_kwargs, seed=agent_seed, **a2c_params)
    elif uncertainty_method == "Dropout":
        model = A2C(DropoutActorCriticPolicy, env, verbose=0 if debug else 1, policy_kwargs=yaml_policy_kwargs, seed=agent_seed, **a2c_params)

    return env, model


def initialize_ddpg_agent(environment_name: str, agent_seed: int, uncertainty_method: str, ensemble_size: int, debug: bool) -> tuple:
    """
    Initializes a DDPG agent using hyperparameters from a YAML configuration file.
    This method loads default DDPG parameters, overrides them with environment-specific settings from a YAML file,
    sets up the environment (with optional normalization and feature extraction settings), and initializes the DDPG model.

    :param environment_name: Name of the Gymnasium environment to train on.
    :param agent_seed: Random seed for reproducibility.
    :param uncertainty_method: The uncertainty estimation method to use.
    :param ensemble_size: The size of the ensemble (if applicable).
    :param debug: Whether to enable verbose output.
    :return: A tuple (environment, DDPG model) where the environment is the configured Gymnasium environment
             and the DDPG model is the initialized RL agent.
    """
    # Create Environment
    env = create_environment(environment_name, agent_seed)

    # Load Default DDPG Parameters
    ddpg_defaults = {k: v.default for k, v in inspect.signature(DDPG.__init__).parameters.items() if v.default is not inspect.Parameter.empty}

    # Load environment-specific YAML Hyperparameters
    with open("hyperparams/ddpg.yaml", "r") as file:
        hyperparams = yaml.safe_load(file)
    env_hyperparams = hyperparams.get(environment_name, {})

    # Merge the DDPG default parameters with the environment-specific YAML parameters.
    ddpg_params = {**ddpg_defaults, **env_hyperparams}
    if "noise_type" in ddpg_params and "noise_std" in ddpg_params:
        n_actions = env.action_space.shape[0]
        mean, sigma = np.zeros(n_actions), ddpg_params["noise_std"] * np.ones(n_actions)
        if ddpg_params["noise_type"] == "normal":
            ddpg_params["action_noise"] = NormalActionNoise(mean=mean, sigma=sigma)
        elif ddpg_params["noise_type"] == "ornstein-uhlenbeck":
            ddpg_params["action_noise"] = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

    # Process Policy Keyword Arguments (policy_kwargs)
    yaml_policy_kwargs = env_hyperparams.get("policy_kwargs", "{}")
    if isinstance(yaml_policy_kwargs, str):
        yaml_policy_kwargs = eval(yaml_policy_kwargs)
    if ddpg_params.get("policy") == "CnnPolicy":
        yaml_policy_kwargs["features_extractor_class"] = NatureCNN

    # Remove Unnecessary Parameters From DDPG-Parameters
    for key in ["policy", "policy_kwargs", "verbose", "seed", "noise_type", "noise_std"]:
        ddpg_params.pop(key, None)

    # Initialize PPO Model
    model = None
    if uncertainty_method == "Ensemble":
        model = EnsembleDDPG(EnsembleDDPGPolicy, env, verbose=0 if debug else 1, policy_kwargs=yaml_policy_kwargs, seed=agent_seed, **ddpg_params)
    if uncertainty_method == "Dropout":
        model = DDPG(DropoutDDPGPolicy, env, verbose=0 if debug else 1, policy_kwargs=yaml_policy_kwargs, seed=agent_seed, **ddpg_params)

    return env, model


def initialize_agent(agent_name: str, environment_name: str, agent_seed: int, uncertainty_method: str, ensemble_size: int, debug: bool) -> tuple:
    env, model = None, None
    if agent_name == "PPO":
        env, model = initialize_ppo_agent(environment_name, agent_seed, uncertainty_method, ensemble_size, debug)
    elif agent_name == "A2C":
        env, model = initialize_a2c_agent(environment_name, agent_seed, uncertainty_method, ensemble_size, debug)
    elif agent_name == "DDPG":
        env, model = initialize_ddpg_agent(environment_name, agent_seed, uncertainty_method, ensemble_size, debug)
    return env, model


def plot_uncertainties(head: str, uncertainty: str, plots_dir: str, interval: int, values: list[list[float]], colors: list):
    """
    Plots the uncertainty values for multiple runs and saves each seed separately.

    :param head: "policy" or "value" indicating which network's uncertainty is plotted.
    :param uncertainty: "aleatoric" or "epistemic" indicating the type of uncertainty.
    :param plots_dir: Base directory where the plots should be saved.
    :param interval: Interval at which uncertainty is measured.
    :param values: List of lists containing uncertainty values across multiple runs.
    :param colors: List of colors for each seed run.
    """
    for i, run_values in enumerate(values):
        seed_dir = os.path.join(plots_dir, f"Seed_{i+1}")
        os.makedirs(seed_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        steps = np.arange(0, len(run_values) * interval, interval)
        plt.plot(steps, run_values, label=f"Run {i+1}", color=colors[i])
        if head == "value":
            plt.yscale("log")  # Use log scale to handle the few extreme output outliers of the value function
        plt.xlabel("Training Steps")
        plt.ylabel("Uncertainty")
        plt.title(f"{uncertainty.capitalize()} Uncertainty in {head.capitalize()} Network (Seed {i+1})")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(seed_dir, f"{uncertainty}_uncertainty_{head}.png"))
        plt.close()


def compute_and_plot_uncertainties(interval: int, plots_dir: str, aus_policy: list[list[float]], eus_policy: list[list[float]], eus_value: list[list[float]]):
    """
    Plots aleatoric and epistemic uncertainties for policy and value networks over the training steps for each seed-run.

    :param interval: Interval at which uncertainty is measured.
    :param plots_dir: Directory where the plots should be saved.
    :param aus_policy: List of lists containing aleatoric uncertainty values for the policy across multiple runs.
    :param eus_policy: List of lists containing epistemic uncertainty values for the policy across multiple runs.
    :param eus_value: List of lists containing epistemic uncertainty values for the value network across multiple runs.
    """
    colors = list(matplotlib.colors.TABLEAU_COLORS.values())[:len(eus_policy)]
    if aus_policy:
        plot_uncertainties("policy", "aleatoric", plots_dir, interval, aus_policy, colors)
    plot_uncertainties("policy", "epistemic", plots_dir, interval, eus_policy, colors)
    plot_uncertainties("value", "epistemic", plots_dir, interval, eus_value, colors)


def plot_rewards(all_runs_rewards: list[list[tuple[int, float]]], plots_dir: str):
    """
    Plots the reward progression over training steps for multiple seed runs.

    :param all_runs_rewards: List of lists containing (Timestep, Reward) tuples for multiple seed runs.
    :param plots_dir: Base directory where the plots should be saved.
    """
    colors = list(matplotlib.colors.TABLEAU_COLORS.values())[:len(all_runs_rewards)]
    for i, run_rewards in enumerate(all_runs_rewards):
        seed_dir = os.path.join(plots_dir, f"Seed_{i+1}")
        os.makedirs(seed_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        timesteps, rewards = zip(*run_rewards)
        plt.plot(timesteps, rewards, label=f"Run {i+1}", color=colors[i])
        plt.xlabel("Training Steps")
        plt.ylabel("Reward")
        plt.title(f"Training Rewards (Seed {i+1})")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(seed_dir, "Rewards.png"))
        plt.close()
