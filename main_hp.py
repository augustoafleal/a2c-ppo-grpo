import argparse
import json
import torch
import copy
import itertools
from train.train_grpo import train_grpo
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
args = parser.parse_args()

with open(args.config, "r") as f:
    base_hp = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_grid = {
    "use_kl": [True, False],
    "grpo_mc": [True, False],
}

keys, values = zip(*hyper_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for i, combo in enumerate(combinations):
    combo["run_id"] = i

csv_filename = "hyperparameter_combinations.csv"
with open(csv_filename, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run_id"] + list(hyper_grid.keys()))
    writer.writeheader()
    for combo in combinations:
        writer.writerow(combo)

runs_per_combo = 10
total_runs = len(combinations) * runs_per_combo
current_run = 0

for combo_id, combo in enumerate(combinations):

    for repeat in range(runs_per_combo):
        current_run += 1

        print(
            f"\n[INFO] Starting run {current_run}/{total_runs} - "
            f"combo {combo_id} (repeat {repeat + 1}/{runs_per_combo}): {combo}"
        )

        hp = copy.deepcopy(base_hp)
        hp.update(combo)
        hp["seed"] = int(np.random.randint(0, 2**31 - 1))
        hp["run_id"] = f"{combo_id}_rep{repeat + 1}"
        print(f"[HP] Hyperparameters for this run: {hp}")

        if hp["agent_type"] in ("grpo", "grpo_batch"):
            train_grpo(hp, device)
        else:
            raise ValueError(f"Agent type {hp['agent_type']} not supported.")

        print(f"[INFO] Finished run {current_run}/{total_runs}")
