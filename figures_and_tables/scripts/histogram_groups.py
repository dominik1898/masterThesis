import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import os

df = pd.read_excel("full_correlations.xlsx")
metric_groups = {
    "tus_policy": [col for col in df.columns if col.startswith("tus_policy")],
    "eus_policy": [col for col in df.columns if col.startswith("eus_policy")],
    "eus_value": [col for col in df.columns if col.startswith("eus_value")]
}

solved_levels = ["green", "yellow", "orange", "red", "brown"]
color_map = {"green": "#2ca02c", "yellow": "#ffdd57", "orange": "#ff7f0e", "red": "#d62728", "brown": "#8B4513"}
log_bins = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]
bin_labels = ["0–0.0001", "0.0001–0.001", "0.001–0.01", "0.01–0.1", "0.1–1", "1–10", "10–100", "100–1000", "1000–10000", ">10000"]
output_folder = "grouped_histograms"
os.makedirs(output_folder, exist_ok=True)


def plot_histogram(ax, df, column):
    clean_df = df[[column, "Solved"]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean_df.empty:
        ax.set_title(f"{column}\n(no data)")
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
    labels_filtered = [labels[i] for i in non_empty]
    x = np.arange(len(labels_filtered))
    bar_width = 0.9 / len(solved_levels)
    for i in range(len(x)):
        color = "#cccccc" if i % 2 == 0 else "#ffffff"
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.9)
    for idx, level in enumerate(solved_levels):
        offsets = x + (idx - len(solved_levels) / 2 + 0.5) * bar_width
        heights = [counts[level][i] for i in non_empty]
        ax.bar(offsets, heights, width=bar_width, color=color_map[level], label=level)

    ax.set_xticks(range(len(labels_filtered)))
    ax.set_xticklabels(labels_filtered, fontsize=7, rotation=45)
    ax.set_xlim(-0.5, len(labels_filtered) - 0.5)
    ax.set_title(column, fontsize=12)
    ax.tick_params(axis='both', labelsize=7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.set_ylim(bottom=0)


def generate_grouped_pdf(df, group_name, columns):
    pdf_path = os.path.join(output_folder, f"histograms_{group_name}.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, axs = plt.subplots(2, 2, figsize=(11.7, 8.3))  # A4 landscape
        axs = axs.flatten()
        fig.subplots_adjust(wspace=0.25, hspace=0.35)
        fig_width, fig_height = fig.get_size_inches()
        fig_width *= fig.dpi
        fig_height *= fig.dpi
        fig.canvas.draw()
        fig.lines.append(plt.Line2D([0.5, 0.5], [0, 1], transform=fig.transFigure, linestyle='--', color='black', linewidth=2))
        fig.lines.append(plt.Line2D([0, 1], [0.5, 0.5], transform=fig.transFigure, linestyle='--', color='black', linewidth=2))
        for i, column in enumerate(columns[:4]):
            plot_histogram(axs[i], df, column)
            axs[i].grid(axis='y', linestyle='--', alpha=0.5)
            axs[i].grid(axis='x', visible=False)
        for j in range(2):
            axs[j + 2].axhline(y=-0.5, color='black', linestyle='--', linewidth=0.8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


for group_name, group_columns in metric_groups.items():
    generate_grouped_pdf(df, group_name, group_columns)
