import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import os

df = pd.read_excel("full_correlations.xlsx")
plot_config = {("A2C", "Dropout"): "eus_policy_rel_std", ("A2C", "Ensemble"): "eus_policy_rel_std",
               ("DDPG", "Dropout"): "eus_value_rel_std", ("DDPG", "Ensemble"): "eus_value_rel_mean",
               ("PPO", "Dropout"): "eus_value_rel_std", ("PPO", "Ensemble"): "eus_policy_rel_std"}
solved_levels = ["green", "yellow", "orange", "red", "brown"]
color_map = {"green": "#2ca02c", "yellow": "#ffdd57", "orange": "#ff7f0e", "red": "#d62728", "brown": "#8B4513"}
log_bins = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]
bin_labels = ["0–0.0001", "0.0001–0.001", "0.001–0.01", "0.01–0.1", "0.1–1", "1–10", "10–100", "100–1000", "1000–10000", ">10000"]

def plot_histogram(ax, data, column, title):
    clean_df = data[[column, "Solved"]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean_df.empty:
        ax.set_title(f"{title}\n(no data)")
        return
    last_bin_mask = clean_df[column] > 10000
    if last_bin_mask.sum() > 7:
        bins = log_bins[:-1] + [100000, 1000000, np.inf]
        labels = bin_labels[:-1] + ["10000–100000", "100000–1000000", ">1000000"]
    else:
        bins = log_bins
        labels = bin_labels
    counts = {level: np.histogram(clean_df[clean_df["Solved"] == level][column], bins=bins)[0] for level in solved_levels}
    total = np.sum([counts[level] for level in solved_levels], axis=0)
    non_empty = np.where(total > 0)[0]
    bins_filtered = [bins[i] for i in non_empty] + [bins[non_empty[-1] + 1]]
    labels_filtered = [labels[i] for i in non_empty]
    x = np.arange(len(labels_filtered))
    bar_width = 0.9 / len(solved_levels)

    for i in range(len(x)):
        bg_color = "#cccccc" if i % 2 == 0 else "#ffffff"
        ax.axvspan(i - 0.5, i + 0.5, color=bg_color, alpha=0.9)
    for idx, level in enumerate(solved_levels):
        offsets = x + (idx - len(solved_levels) / 2 + 0.5) * bar_width
        heights = [counts[level][i] for i in non_empty]
        ax.bar(offsets, heights, width=bar_width, color=color_map[level])

    ax.set_xticks(range(len(labels_filtered)))
    ax.set_xticklabels(labels_filtered, fontsize=7, rotation=45)
    ax.set_xlim(-0.5, len(labels_filtered) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', labelsize=7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.grid(axis='x', visible=False)


with PdfPages("figures/selected_histograms.pdf") as pdf:
    fig, axs = plt.subplots(3, 2, figsize=(11.7, 8.3))
    axs = axs.flatten()
    for i, ((agent, uq), metric) in enumerate(plot_config.items()):
        subset = df[(df["Agent"] == agent) & (df["Uncertainty Method"] == uq)]
        plot_title = f"{agent} - {uq}        ({metric})"
        plot_histogram(axs[i], subset, metric, plot_title)
    for ax in axs[len(plot_config):]:
        ax.axis('off')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.25, hspace=0.6)
    fig.lines.append(plt.Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, linestyle='--', color='black', linewidth=2.0))
    fig.lines.append(plt.Line2D([0, 1], [2 / 3, 2 / 3], transform=fig.transFigure, linestyle='--', color='black', linewidth=2.0))
    fig.lines.append(plt.Line2D([0, 1], [1 / 3, 1 / 3], transform=fig.transFigure, linestyle='--', color='black', linewidth=2.0))
    pdf.savefig(fig)
    plt.close(fig)
