import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_excel("full_correlations.xlsx")
reward_cols = [col for col in df.columns if "rewards" in col]
uncertainty_cols = [col for col in df.columns if ("policy" in col or "value" in col)]
metric_cols = reward_cols + uncertainty_cols
ratio_cols = [col for col in df.columns if col.startswith("ratio_")]
metric_cols += ratio_cols
solved_levels = ["green", "yellow", "orange", "red", "brown"]
color_map = {"green": "#2ca02c", "yellow": "#ffdd57", "orange": "#ff7f0e", "red": "#d62728", "brown": "#8B4513"}
log_bins = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]
bin_labels = ["0–0.0001", "0.0001–0.001", "0.001–0.01", "0.01–0.1", "0.1–1", "1–10", "10–100", "100–1000", "1000–10000", ">10000"]
output_folder = "histograms"
os.makedirs(output_folder, exist_ok=True)
subgroup_columns = ["Agent", "Uncertainty Method", "Observation Space", "Action Space", "Transition", "Reward Density"]


def plot_metric_histogram(data_frame, column, save_path):
    clean_df = data_frame[[column, "Solved"]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean_df.empty:
        print(f"Skipping {column} (no valid data in this subset).")
        return

    last_bin_mask = clean_df[column] > 10000
    if last_bin_mask.sum() > 7:
        extended_bins = log_bins[:-1] + [100000, 1000000, np.inf]
        extended_labels = bin_labels[:-1] + ["10000–100000", "100000–1000000", ">1000000"]
    else:
        extended_bins = log_bins
        extended_labels = bin_labels

    bin_counts = {level: np.histogram(clean_df[clean_df["Solved"] == level][column], bins=extended_bins)[0] for level in solved_levels}
    total_counts = np.sum([bin_counts[level] for level in solved_levels], axis=0)
    non_empty_indices = np.where(total_counts > 0)[0]
    filtered_bins = [extended_bins[i] for i in non_empty_indices] + [extended_bins[non_empty_indices[-1] + 1]]
    filtered_labels = [extended_labels[i] for i in non_empty_indices]
    x = np.arange(len(filtered_labels))
    bin_counts = {level: [bin_counts[level][i] for i in non_empty_indices] for level in solved_levels}

    plt.figure(figsize=(9, 6))

    num_levels = len(solved_levels)

    for i in range(len(x)):
        color = "#999999" if i % 2 == 0 else "white"
        plt.axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.5, zorder=0)

    bar_width = 0.9 / num_levels

    for i, level in enumerate(solved_levels):
        offsets = x + (i - num_levels / 2 + 0.5) * bar_width
        plt.bar(offsets, bin_counts[level], width=bar_width, color=color_map[level], label=level)

    plt.xticks(ticks=range(len(filtered_bins) - 1), labels=filtered_labels, fontsize=8)

    plt.title(f"Histogram of '{column}'", fontsize=13)
    plt.xlabel("Binned Value Range", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{column.replace('/', '_')}.png"))
    plt.close()


def generate_all_histograms(data_frame, base_output_folder, group_label=None, group_value=None):
    data_frame = data_frame.copy()
    data_frame = data_frame[data_frame["Solved"].notna()]
    data_frame["Solved"] = data_frame["Solved"].astype(str)
    if group_label and group_value:
        save_path = os.path.join(base_output_folder, group_label, str(group_value))
    else:
        save_path = base_output_folder
    for col in metric_cols:
        plot_metric_histogram(data_frame, col, save_path)


mujoco_envs = {"Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"}

def classify_space(row, space_type):
    env = row["Environment"]
    original_type = row[space_type]
    if original_type == "Discrete":
        return "Discrete"
    elif env in mujoco_envs:
        return "MuJoCo"
    else:
        return "Non-MuJoCo-Continuous"

df["Observation Space"] = df.apply(lambda row: classify_space(row, "Observation Space"), axis=1)
df["Action Space"] = df.apply(lambda row: classify_space(row, "Action Space"), axis=1)

generate_all_histograms(df, output_folder)
for group_col in subgroup_columns:
    for value in df[group_col].dropna().unique():
        subset_df = df[df[group_col] == value]
        generate_all_histograms(subset_df, output_folder, group_label=group_col, group_value=value)

for agent in df["Agent"].dropna().unique():
    for method in df["Uncertainty Method"].dropna().unique():
        subset_df = df[(df["Agent"] == agent) & (df["Uncertainty Method"] == method)]
        if not subset_df.empty:
            group_name = f"{agent}_{method}"
            generate_all_histograms(subset_df, output_folder, group_label="Agent+UQ", group_value=group_name)
