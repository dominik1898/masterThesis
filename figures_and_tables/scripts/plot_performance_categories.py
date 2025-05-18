import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

examples = {
    "Green": {"path": "results/CartPole/PPO/Ensemble", "seed": 2, "threshold": 475.0, "color": "green"},
    "Yellow": {"path": "results/Reacher/PPO/Dropout", "seed": 3, "threshold": -3.75, "color": "gold"},
    "Orange": {"path": "results/BipedalWalker/A2C/Ensemble", "seed": 1, "threshold": 300, "color": "orange"},
    "Red": {"path": "results/Pusher/DDPG/Ensemble", "seed": 2, "threshold": 0.0, "color": "red"},
    "Brown": {"path": "results/Hopper/PPO/Dropout", "seed": 3, "threshold": 3800.0, "color": "saddlebrown"},
}

fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1, 1], figure=fig)
axes = []
axes.append(fig.add_subplot(gs[0, 1:3]))
axes.append(fig.add_subplot(gs[0, 3:5]))
axes.append(fig.add_subplot(gs[1, 0:2]))
axes.append(fig.add_subplot(gs[1, 2:4]))
axes.append(fig.add_subplot(gs[1, 4:6]))

for idx, (category, info) in enumerate(examples.items()):
    ax = axes[idx]
    threshold = info["threshold"]
    data = np.load(os.path.join(info["path"], "rewards.npy"), allow_pickle=True)
    run_data = data[info["seed"] - 1]
    steps = [step for (step, reward) in run_data]
    rewards = [reward for (step, reward) in run_data]
    ax.plot(steps, rewards, label="Reward", linewidth=1.5)
    ax.axhline(y=threshold, color="black", linestyle="--", linewidth=2.0, label="Threshold")
    env = info["path"].split('/')[1]
    agent = info["path"].split('/')[2]
    method = info["path"].split('/')[3]
    main_title = f"{env} • {agent} • {method} • Seed {info['seed']}"
    title = f"{main_title.ljust(50)}Reward Threshold: {threshold}"
    ax.set_title(title, fontsize=10, loc="center")
    ax.grid(True, linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(info['color'])
        spine.set_linewidth(5)

plt.tight_layout()
plt.savefig("./all_categories_overview.pdf", dpi=600)
plt.close()
