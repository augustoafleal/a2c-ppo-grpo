import argparse
import json
import torch
import evaluate
from train.train_a2c import train_a2c
from train.train_grpo import train_grpo
from train.train_ppo import train_ppo

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
args = parser.parse_args()

with open(args.config, "r") as f:
    hp = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if hp.get("load_model"):
    print(f"[INFO] Evaluation mode. Using model from {hp['load_model']}")
    evaluate.run(hp, device)
else:
    print(f"[INFO] Training mode with agent type: {hp['agent_type']}.")
    if hp["agent_type"] in ("grpo", "grpo_batch"):
        train_grpo(hp, device)
    elif hp["agent_type"] == "ppo":
        train_ppo(hp, device)
    else:
        train_a2c(hp, device)
