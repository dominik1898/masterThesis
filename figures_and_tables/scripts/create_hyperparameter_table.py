import yaml
import os
from collections import defaultdict

environments = ["Acrobot", "CartPole", "MountainCarContinuous", "MountainCar", "Pendulum", "BipedalWalker", "CarRacing", "LunarLander", "Blackjack", "Taxi", "CliffWalking", "FrozenLake",
                "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher", "Swimmer", "Walker2d"]
agents = ["A2C", "PPO", "DDPG"]

full_configs = defaultdict(dict)

for file_name in os.listdir("../../hyperparams"):
    agent_name = file_name.split('.')[0].upper()
    with open(os.path.join("../../hyperparams", file_name), 'r') as f:
        config = yaml.safe_load(f)

    for env, params in config.items():
        if isinstance(params, str):
            full_configs[agent_name][env] = params
        else:
            full_configs[agent_name][env] = params

with open("figures/hyperparams_table.tex", 'w') as f:
    f.write("\\setlength{\\tabcolsep}{8pt}\n")
    f.write("\\renewcommand{\\arraystretch}{1.2}\n")
    f.write("\\rowcolors{2}{blue!5}{white}\n")
    f.write("\\begin{longtable}{|>{\\raggedright\\arraybackslash}p{3.5cm}|" + "|".join([">{\\raggedright\\arraybackslash}p{4cm}" for _ in agents]) + "|}\n")
    f.write("\\caption{Xy}\n")
    f.write("\\hline\n")
    f.write("\\rowcolor{blue!20}\n")
    header = "Environment & " + " & ".join(agents) + " \\\\ \n"
    f.write(header)
    f.write("\\hline\n")
    f.write("\\endfirsthead\n")
    f.write("\\hline\n")
    f.write("\\rowcolor{blue!20}\n")
    f.write(header)
    f.write("\\hline\n")
    f.write("\\endhead\n")

    for env in environments:
        """param_str = "\\scriptsize " + " \\\\ ".join(param_lines)"""
        row = [env]
        for agent in agents:
            params = full_configs.get(agent, {}).get(env, "")
            if isinstance(params, dict):
                param_lines = []
                for k, v in params.items():
                    safe_key = k.replace('_', '\\_')
                    safe_value = str(v).replace('_', '\\_')
                    param_lines.append(f"{safe_key}: {safe_value}")
                param_str = ("\\scriptsize \\begin{tabular}[t]{@{}l@{}}\n" + " \\\\\n".join(param_lines) + "\n\\end{tabular}")
            else:
                param_str = ""
            row.append(param_str)
        line = " & ".join(row) + " \\\\ \n"
        f.write(line)
        f.write("\\hline\n")

    f.write("\\end{longtable}\n")
    f.write("\\label{tab:hyperparams_table}\n")
