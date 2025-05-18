import pandas as pd
import numpy as np

df = pd.read_excel("full_correlations.xlsx")
df = df[df["Solved"].isin(["red", "brown"])]

metrics = ["tus_policy_end_mean", "tus_policy_rel_mean", "eus_policy_end_mean", "eus_policy_rel_mean", "eus_value_end_mean", "eus_value_rel_mean",
           "ratio_end_tus_ep", "ratio_end_tus_ev", "ratio_end_ep_ev", "ratio_rel_tus_ep", "ratio_rel_tus_ev", "ratio_rel_ep_ev"]

def summarize_group(group, group_keys):
    results = []
    for group_values, df_group in group:
        for color in ["red", "brown"]:
            color_subset = df_group[df_group["Solved"] == color]
            summary = dict(zip(group_keys, group_values if isinstance(group_values, tuple) else (group_values,)))
            summary["Solved"] = color
            for metric in metrics:
                values = pd.to_numeric(color_subset[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) == 0:
                    mean, std = "", ""
                else:
                    mean = round(values.mean(), 3)
                    std = round(values.std(ddof=1), 3)
                    if abs(mean) >= 1000:
                        mean = "{:.1e}".format(mean).replace("e+", "e")
                    if abs(std) >= 1000:
                        std = "{:.1e}".format(std).replace("e+", "e")
                summary[metric + "_mean"] = mean
                summary[metric + "_std"] = std
            results.append(summary)
    return pd.DataFrame(results)

grouped_by_agent = df.groupby("Agent")
agent_df = summarize_group(grouped_by_agent, ["Agent"])
agent_df.to_excel("uq_summary_by_agent.xlsx", index=False)

grouped_by_method = df.groupby("Uncertainty Method")
method_df = summarize_group(grouped_by_method, ["Uncertainty Method"])
method_df.to_excel("uq_summary_by_uqmethod.xlsx", index=False)

global_results = []
for color in ["red", "brown"]:
    color_subset = df[df["Solved"] == color]
    row = {"Solved": color}
    for metric in metrics:
        values = pd.to_numeric(color_subset[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) == 0:
            mean, std = "", ""
        else:
            mean = round(values.mean(), 3)
            std = round(values.std(ddof=1), 3)
        row[metric + "_mean"] = mean
        row[metric + "_std"] = std
    global_results.append(row)

global_df = pd.DataFrame(global_results)
global_df.to_excel("uq_summary_global_red_brown.xlsx", index=False)
