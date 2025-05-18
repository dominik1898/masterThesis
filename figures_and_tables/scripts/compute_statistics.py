import os
import numpy as np
import pandas as pd

agents = ["A2C", "PPO", "DDPG"]
methods = ["Dropout", "Ensemble"]
environments = {
    "A2C": ["Acrobot", "CartPole", "MountainCarContinuous", "MountainCar", "Pendulum", "BipedalWalker", "CarRacing", "LunarLander", "Blackjack", "Taxi", "CliffWalking", "FrozenLake",
            "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"],
    "PPO": ["Acrobot", "CartPole", "MountainCarContinuous", "MountainCar", "Pendulum", "BipedalWalker", "CarRacing", "LunarLander", "Blackjack", "Taxi", "CliffWalking", "FrozenLake",
            "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"],
    "DDPG": ["MountainCarContinuous", "Pendulum", "BipedalWalker", "CarRacing", "LunarLander",
             "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"]}
files = ["rewards", "aus_policy", "eus_policy", "eus_value"]
default_value = np.inf

def compute_stats(values, skip_start_pct=0.0):
    if len(values) < 2:
        return [default_value] * 4
    start_idx = int(len(values) * skip_start_pct)
    start_end = int(len(values) * (skip_start_pct + 0.05))
    end_start = int(len(values) * 0.95)
    start_slice, end_slice = values[start_idx:start_end], values[end_start:]
    if len(start_slice) < 5:
        start_slice = values[:5]
        end_slice = values[-5:]
    start_mean, start_std, end_mean, end_std = np.mean(start_slice), np.std(start_slice), np.mean(end_slice), np.std(end_slice)

    if start_mean == 0 or start_std == 0:
        extended_end = start_end + len(start_slice)
        extended_slice = values[start_idx:extended_end]
        if len(extended_slice) >= 5:
            start_mean = np.mean(extended_slice)
            start_std = np.std(extended_slice)

    rel_mean = end_mean / start_mean if start_mean != 0 else default_value
    rel_std = end_std / start_std if start_std != 0 else default_value
    return end_mean, end_std, rel_mean, rel_std

def process_file(file_path, file_type):
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception:
        return [default_value] * 4

    all_runs_stats = []
    for run in data:
        if file_type == "rewards":
            rewards = [r for (_, r) in run]
            stats = compute_stats(rewards)
        elif file_type == "eus_policy":
            stats = compute_stats(run, skip_start_pct=0.05)
        else:
            stats = compute_stats(run)
        all_runs_stats.append(stats)

    try:
        mean_stats = np.mean(all_runs_stats, axis=0)
        return list(mean_stats)
    except:
        return [default_value] * 4

def safe_div(a, b):
    return a / b if b != 0 and not np.isinf(b) else default_value

results = []
for agent in agents:
    for method in methods:
        for env in environments[agent]:
            row_id = f"{agent}/{method}/{env}"
            row = {"ID": row_id}
            for key in files:
                if agent == "DDPG" and key == "aus_policy":
                    row.update({f"{key}_{suffix}": default_value for suffix in ["end_mean", "end_std", "rel_mean", "rel_std"]})
                    continue
                path = os.path.join("../../results", env, agent, method, f"{key}.npy")
                stats = process_file(path, key)
                row.update({f"{key}_end_mean": stats[0], f"{key}_end_std": stats[1], f"{key}_rel_mean": stats[2], f"{key}_rel_std": stats[3]})
            a_end_mean = row.get("aus_policy_end_mean", default_value)
            ep_end_mean = row.get("eus_policy_end_mean", default_value)
            ev_end_mean = row.get("eus_value_end_mean", default_value)
            a_rel_mean = row.get("aus_policy_rel_mean", default_value)
            ep_rel_mean = row.get("eus_policy_rel_mean", default_value)
            ev_rel_mean = row.get("eus_value_rel_mean", default_value)
            row["ratio_end_tus_ep"] = safe_div(a_end_mean, ep_end_mean)
            row["ratio_end_tus_ev"] = safe_div(a_end_mean, ev_end_mean)
            row["ratio_end_ep_ev"] = safe_div(ep_end_mean, ev_end_mean)
            row["ratio_rel_tus_ep"] = safe_div(a_rel_mean, ep_rel_mean)
            row["ratio_rel_tus_ev"] = safe_div(a_rel_mean, ev_rel_mean)
            row["ratio_rel_ep_ev"] = safe_div(ep_rel_mean, ev_rel_mean)
            results.append(row)
df = pd.DataFrame(results).round(7)
df.drop(columns=[col for col in ["rewards_end_mean", "rewards_rel_mean"] if col in df.columns], inplace=True)

split_df = df["ID"].str.split("/", expand=True)
split_df.columns = ["Agent", "Uncertainty Method", "Environment"]
df = pd.concat([split_df, df.drop(columns=["ID"])], axis=1)

basic_df = pd.read_excel("Correlations.xlsx")
merged_df = pd.merge(basic_df, df, on=["Agent", "Uncertainty Method", "Environment"], how="inner")
rename_dict = {col: col.replace("aus_policy", "tus_policy") for col in merged_df.columns if col.startswith("aus_policy")}
merged_df = merged_df.rename(columns=rename_dict)
merged_df.to_excel("full_correlations.xlsx", index=False)
