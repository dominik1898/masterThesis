import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

AGENTS = ["A2C", "PPO", "DDPG"]
UQ_METHODS = ["Dropout", "Ensemble"]
SEEDS = [0, 1, 2]

SORTED_ENVIRONMENTS = {
    "A2C": ["Acrobot", "CartPole", "MountainCarContinuous", "MountainCar", "Pendulum", "BipedalWalker", "CarRacing", "LunarLander", "Blackjack", "Taxi", "CliffWalking", "FrozenLake",
            "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"],
    "PPO": ["Acrobot", "CartPole", "MountainCarContinuous", "MountainCar", "Pendulum", "BipedalWalker", "CarRacing", "LunarLander", "Blackjack", "Taxi", "CliffWalking", "FrozenLake",
            "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"],
    "DDPG": ["MountainCarContinuous", "Pendulum", "BipedalWalker", "CarRacing", "LunarLander",
             "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"]
}

STEP_INTERVAL = 2048
BASE_PATH = "../../results"
OUTPUT_PDF = "uncertainty_and_reward_plots.pdf"

def load_npy(path):
    return np.load(path, allow_pickle=True)

def prepare_uncertainty_x(length):
    return np.arange(length) * STEP_INTERVAL

def prepare_reward_x_y(data):
    x = [entry[0] for entry in data]
    y = [entry[1] for entry in data]
    return np.array(x), np.array(y)

def running_average(y, window=50):
    smoothed = []
    for i in range(len(y)):
        start = 0 if i < window else i - window + 1
        smoothed.append(np.mean(y[start:i + 1]))
    return np.array(smoothed)

def plot_subplot(ax, x, y, title=None, log=False):
    if y is None or len(y) == 0:
        ax.set_visible(False)
        return
    if log:
        ax.semilogy(x, y)
    else:
        ax.plot(x, y)
    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True)


with PdfPages(OUTPUT_PDF) as pdf:
    for agent in AGENTS:
        for env in SORTED_ENVIRONMENTS[agent]:
            n_columns = 4 if agent != "DDPG" else 3
            fig, axs = plt.subplots(nrows=6, ncols=n_columns, figsize=(n_columns * 3.2, 6 * 2))
            fig.suptitle(f"{agent} • {env}", fontsize=22)
            column_titles = ["Reward"]
            if agent != "DDPG":
                column_titles += ["Policy-TU"]
            column_titles += ["Policy-EU", "Value-EU"]
            for i, title in enumerate(column_titles):
                fig.text(x=(i + 0.5) / n_columns, y=0.92, s=title, fontsize=10, fontweight="bold", ha='center')
            for uq_idx, uq_method in enumerate(UQ_METHODS):
                for seed_idx in range(3):
                    row = uq_idx * 3 + seed_idx
                    folder = os.path.join(BASE_PATH, env, agent, uq_method)
                    rewards_all = load_npy(os.path.join(folder, "rewards.npy"))
                    eus_value_all = load_npy(os.path.join(folder, "eus_value.npy"))
                    eus_policy_all = load_npy(os.path.join(folder, "eus_policy.npy"))
                    aus_policy_all = load_npy(os.path.join(folder, "aus_policy.npy")) if agent != "DDPG" else [None]*3
                    reward_x, reward_y = prepare_reward_x_y(rewards_all[seed_idx])
                    if env in ["Blackjack", "FrozenLake"]:
                        reward_y = running_average(reward_y, window=50)
                    uq_x = prepare_uncertainty_x(len(eus_value_all[seed_idx]))
                    col = 0
                    plot_subplot(axs[row, col], reward_x, reward_y, title=f"{uq_method} – Seed {seed_idx + 1}"); col += 1
                    if agent != "DDPG":
                        plot_subplot(axs[row, col], uq_x, aus_policy_all[seed_idx]); col += 1
                    plot_subplot(axs[row, col], uq_x, eus_policy_all[seed_idx]); col += 1
                    plot_subplot(axs[row, col], uq_x, eus_value_all[seed_idx], log=True)

            plt.subplots_adjust(top=0.92)
            plt.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)
