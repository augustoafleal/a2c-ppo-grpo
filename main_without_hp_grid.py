import argparse
import json
import torch
import copy
from train.train_grpo import train_grpo
from train.train_ppo import train_ppo
from train.train_a2c import train_a2c
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
args = parser.parse_args()

with open(args.config, "r") as f:
    base_hp = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_runs = 1
current_run = 0

for repeat in range(total_runs):
    current_run += 1
    print(f"\n[INFO] Starting run {current_run}/{total_runs}")
    hp = copy.deepcopy(base_hp)
    hp["seed"] = int(np.random.randint(0, 2**31 - 1))
    hp["run_id"] = f"_rep{repeat + 1}"
    print(f"[HP] Hyperparameters for this run: {hp}")
    if hp["agent_type"] in ("grpo", "grpo_batch"):
        train_grpo(hp, device)
    elif hp["agent_type"] == "ppo":
        train_ppo(hp, device)
    elif hp["agent_type"] == "a2c":
        train_a2c(hp, device)
    else:
        raise ValueError(f"Agent type {hp['agent_type']} not supported.")
    print(f"[INFO] Finished run {current_run}/{total_runs}")
