import argparse
import copy
import json
import os
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

def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _run_training(run_hp):
    print(f"[INFO] Training mode with agent type: {run_hp['agent_type']}.")
    if run_hp["agent_type"] in ("grpo", "grpo_batch"):
        train_grpo(run_hp, device)
    elif run_hp["agent_type"] == "ppo":
        train_ppo(run_hp, device)
    else:
        train_a2c(run_hp, device)


sweep = hp.get("sweep", {})
if sweep.get("enabled", False):
    if hp.get("load_model"):
        raise ValueError("Sweep is not supported in evaluation mode. Remove 'load_model' from config.")
    if hp.get("agent_type") != "grpo":
        raise ValueError("Sweep is supported only for GRPO training. Set 'agent_type' to 'grpo'.")

    envs = sweep.get("envs", [hp["env_name"]])
    kl_coefs = sweep.get("kl_coefs", [hp.get("kl_coef", 0.0 if not hp.get("use_kl", False) else hp["kl_coef"])])
    seeds = sweep.get("seeds", [hp.get("seed", 0)])
    log_root = sweep.get("log_root", "logs/kl_sweep")
    run_tag = sweep.get("run_tag", "kl_sweep")

    total_runs = len(envs) * len(kl_coefs) * len(seeds)
    current_run = 0

    for env_name in envs:
        env_safe = _safe_name(env_name)
        for kl_coef in kl_coefs:
            use_kl = float(kl_coef) > 0.0
            kl_tag = str(kl_coef).replace(".", "p")
            for seed in seeds:
                current_run += 1
                run_hp = copy.deepcopy(hp)
                run_hp["env_name"] = env_name
                run_hp["seed"] = int(seed)
                run_hp["use_kl"] = use_kl
                run_hp["kl_coef"] = float(kl_coef)
                run_hp["run_id"] = f"{run_tag}__{env_safe}__kl_{kl_tag}__seed_{seed}"
                run_hp["log_dir"] = os.path.join(log_root, env_safe, f"kl_{kl_coef}", f"seed_{seed}")

                os.makedirs(run_hp["log_dir"], exist_ok=True)
                with open(os.path.join(run_hp["log_dir"], "config.json"), "w") as f:
                    json.dump(run_hp, f, indent=4)

                print(
                    f"\n[INFO] Sweep run {current_run}/{total_runs} | "
                    f"env={env_name} | kl_coef={kl_coef} | seed={seed}"
                )
                _run_training(run_hp)
else:
    if hp.get("load_model"):
        print(f"[INFO] Evaluation mode. Using model from {hp['load_model']}")
        evaluate.run(hp, device)
    else:
        _run_training(hp)
