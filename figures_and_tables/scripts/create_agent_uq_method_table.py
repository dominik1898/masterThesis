import pandas as pd
import numpy as np

df = pd.read_excel("full_correlations.xlsx")

agents = ["A2C", "PPO", "DDPG"]
uq_methods = ["Dropout", "Ensemble"]

metrics = ["tus_policy_end_mean", "tus_policy_end_std", "tus_policy_rel_mean", "tus_policy_rel_std",
           "eus_policy_end_mean", "eus_policy_end_std", "eus_policy_rel_mean", "eus_policy_rel_std",
           "eus_value_end_mean", "eus_value_end_std", "eus_value_rel_mean", "eus_value_rel_std"]

colors = ["green", "yellow", "orange", "red", "brown"]

results = {}

for agent in agents:
    for uq in uq_methods:
        subset = df[(df["Agent"] == agent) & (df["Uncertainty Method"] == uq)]
        key = f"{agent}_{uq}"
        results[key] = {}
        for color in colors:
            color_subset = subset[subset["Solved"] == color]
            metrics_summary = {}
            for metric in metrics:
                values = color_subset[metric]
                valid_values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_values) == 0:
                    mean, std = "", ""
                else:
                    mean = round(valid_values.mean(), 3)
                    std = round(valid_values.std(ddof=1), 3)
                    if abs(mean) >= 1000:
                        mean = "{:.1e}".format(mean).replace("e+", "e")
                    if abs(std) >= 1000:
                        std = "{:.1e}".format(std).replace("e+", "e")
                metrics_summary[metric + "_mean"] = mean
                metrics_summary[metric + "_std"] = std

            results[key][color] = metrics_summary

for key in results:
    output_df = pd.DataFrame.from_dict(results[key], orient="index")
    output_df.index.name = "Solved"
    output_df.to_excel(f"{key}_uncertainty_summary.xlsx")
