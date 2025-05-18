import numpy as np
import matplotlib.pyplot as plt

rewards_dropout = np.load("../../results/Swimmer/A2C/Dropout/rewards.npy", allow_pickle=True)[2]
rewards_ensemble = np.load("../../results/Swimmer/A2C/Ensemble/rewards.npy", allow_pickle=True)[1]
uncert_dropout = np.load("../../results/Swimmer/A2C/Dropout/eus_value.npy", allow_pickle=True)[2]
uncert_ensemble = np.load("../../results/Swimmer/A2C/Ensemble/eus_value.npy", allow_pickle=True)[1]

def extract_reward_xy(data):
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    return x, y

def extract_uncert_xy(data):
    x = [i * 2048 for i in range(len(data))]
    y = data
    return x, y

x_r_drop, y_r_drop = extract_reward_xy(rewards_dropout)
x_r_ens, y_r_ens = extract_reward_xy(rewards_ensemble)
x_u_drop, y_u_drop = extract_uncert_xy(uncert_dropout)
x_u_ens, y_u_ens = extract_uncert_xy(uncert_ensemble)

fig, axs = plt.subplots(2, 2, figsize=(11.7, 8.3))  # A4 quer
axs = axs.flatten()

axs[0].plot(x_r_drop, y_r_drop, color="blue")
axs[0].set_title("Reward                            (Swimmer · A2C · Dropout)")
axs[1].plot(x_r_ens, y_r_ens, color="green")
axs[1].set_title("Reward                            (Swimmer · A2C · Ensemble)")
axs[2].plot(x_u_drop, y_u_drop, color="blue")
axs[2].set_title("Value-EU                            (Swimmer · A2C · Dropout)")
axs[2].set_yscale("log")
axs[3].plot(x_u_ens, y_u_ens, color="green")
axs[3].set_title("Value-EU                            (Swimmer · A2C · Ensemble)")
axs[3].set_yscale("log")

fig.subplots_adjust(wspace=0.3, hspace=0.4)
fig.lines.append(plt.Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, linestyle='--', color='black', linewidth=2))

plt.tight_layout()
plt.savefig("Swimmer_A2C_Comparison.pdf", dpi=600)
plt.close()
