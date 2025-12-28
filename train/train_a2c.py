import os
import time
import torch
import numpy as np
import gymnasium as gym
import gymnasium_robotics
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, FlattenObservation
from A2C import A2C
from util.Logger import Logger
from util.RenderRecorder import RenderRecorder
from util.AtariUtils import FireResetEnv


class FetchGoalErrorWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        obs_space = env.observation_space["observation"]
        goal_space = env.observation_space["desired_goal"]

        low = np.concatenate(
            [
                obs_space.low,
                -np.inf * np.ones(goal_space.shape, dtype=np.float32),
            ]
        )

        high = np.concatenate(
            [
                obs_space.high,
                np.inf * np.ones(goal_space.shape, dtype=np.float32),
            ]
        )

        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        goal_error = obs["desired_goal"] - obs["achieved_goal"]
        return np.concatenate([obs["observation"], goal_error], axis=0).astype(np.float32)


def make_gym_atari_env(env_id, num_envs, seed=0, frame_skip=4, stack_size=4, render_mode=None):
    def make_single_env():
        def _init():
            env = gym.make(
                env_id,
                frameskip=1,
                render_mode=render_mode,
                obs_type="rgb",
                full_action_space=False,
                repeat_action_probability=0,
            )
            env = AtariPreprocessing(
                env,
                frame_skip=frame_skip,
                grayscale_obs=True,
                scale_obs=True,
                terminal_on_life_loss=False,
                noop_max=30,
                screen_size=84,
            )
            env = FireResetEnv(env)
            env = FrameStackObservation(env, stack_size=stack_size)
            return env

        return _init

    return SyncVectorEnv([make_single_env() for _ in range(num_envs)])


def make_classic_env(env_name, num_envs):
    def make_single_env():
        def _init():
            if env_name == "Pinball-v0":
                import envs.pinball

                return gym.make(env_name, config_name="hard")
            elif env_name.startswith("Fetch"):
                env = gym.make(env_name, max_episode_steps=50)
                env = FetchGoalErrorWrapper(env)
                return env
            else:
                return gym.make(env_name)

        return _init

    return SyncVectorEnv([make_single_env() for _ in range(num_envs)])


def train_a2c(hp, device):

    critic_losses, actor_losses, entropies = [], [], []

    if hp["atari_mode"]:
        envs = make_gym_atari_env(
            hp["env_name"],
            num_envs=hp["n_envs"],
            seed=hp["seed"],
            frame_skip=hp["frame_skip"],
            stack_size=hp["stack_size"],
            render_mode=hp.get("render_mode", None),
        )
        obs_space = 84 * 84 * hp["stack_size"]
    else:
        envs = make_classic_env(hp["env_name"], hp["n_envs"])
        obs_space = envs.single_observation_space.shape[0]

    action_space = envs.single_action_space

    if hp["is_continuous_actions"]:
        is_continuous = True
        act_space = action_space.shape[0]
    else:
        is_continuous = False
        act_space = action_space.n

    agent = A2C(
        agent_type=hp["agent_type"],
        n_features=obs_space,
        n_actions=act_space,
        device=device,
        critic_lr=hp["critic_lr"],
        actor_lr=hp["actor_lr"],
        n_envs=hp["n_envs"],
        atari_mode=hp["atari_mode"],
        is_continuous_actions=hp["is_continuous_actions"],
    )

    logger = Logger(
        episode_filename=f"logs/a2c_episodes_{hp['run_id']}.csv",
        update_filename=f"logs/a2c_updates_{hp['run_id']}.csv",
        resources_filename=f"logs/a2c_resources_{hp['run_id']}.csv",
    )

    states, _ = envs.reset(seed=hp["seed"])
    episode_rewards = np.zeros(hp["n_envs"], dtype=np.float32)
    last_episode_rewards = np.zeros(hp["n_envs"], dtype=np.float32)
    worker_episodes = np.zeros(hp["n_envs"], dtype=int)
    worker_timesteps = np.zeros(hp["n_envs"], dtype=int)
    episodes_finished_worker0 = 0

    is_fetch_env = hp["env_name"].startswith("Fetch")
    success_reached = np.zeros(hp["n_envs"], dtype=bool)
    success_step = success_step = [None] * hp["n_envs"]

    total_time_steps = 0
    max_time_steps = hp["max_time_steps"]
    update = 0
    start_time = time.time()
    total_iteration = 0
    max_iteration = hp["max_episodes"]

    if hp["atari_mode"]:
        max_iteration = hp["max_episode_steps"]
    else:
        max_iteration = hp["max_episodes"]
    print(f"Max iteration per env: {max_iteration}")

    while total_iteration < max_iteration:
        update_start_time = time.time()
        update += 1
        worker_timesteps += 1

        rollouts = {
            "value_preds": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], device=device),
            "rewards": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], device=device),
            "old_log_probs": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], device=device),
            "masks": torch.ones(hp["n_steps_per_update"], hp["n_envs"], device=device),
            "entropies": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], device=device),
            "logits": torch.zeros(hp["n_steps_per_update"], hp["n_envs"], agent.n_actions, device=device),
        }

        if hp["atari_mode"]:
            rollouts["states"] = torch.zeros(
                hp["n_steps_per_update"], hp["n_envs"], hp["stack_size"], 84, 84, device=device
            )
        else:
            rollouts["states"] = torch.zeros(hp["n_steps_per_update"], hp["n_envs"], obs_space, device=device)

        if hp["is_continuous_actions"]:
            rollouts["actions"] = torch.zeros(
                hp["n_steps_per_update"], hp["n_envs"], act_space, dtype=torch.float32, device=device
            )

            rollouts["pre_tanh_actions"] = torch.zeros(
                hp["n_steps_per_update"], hp["n_envs"], act_space, dtype=torch.float32, device=device
            )
        else:
            rollouts["actions"] = torch.zeros(hp["n_steps_per_update"], hp["n_envs"], dtype=torch.long, device=device)

        for step in range(hp["n_steps_per_update"]):

            if hp["is_continuous_actions"]:
                actions, action_log_probs, state_value_preds, entropy, raw_actions = agent.select_action(states)
                rollouts["pre_tanh_actions"][step] = raw_actions.detach().to(device=device, dtype=torch.float32)
            else:
                actions, action_log_probs, state_value_preds, entropy, logits = agent.select_action(states)
                rollouts["logits"][step] = logits.detach()

            actions_to_env = actions.detach().cpu().numpy()

            if hp["is_continuous_actions"]:
                actions_to_env = np.clip(actions_to_env, action_space.low, action_space.high)

            next_states, rewards, terminated, truncated, infos = envs.step(actions_to_env)

            if is_fetch_env:
                is_success = infos.get("is_success", None)
                if is_success is not None:
                    for i in range(hp["n_envs"]):
                        if (not success_reached[i]) and is_success[i]:
                            success_reached[i] = True
                            success_step[i] = worker_timesteps[i]

            total_time_steps += hp["n_envs"]

            episode_rewards += rewards
            if hp["atari_mode"]:
                rewards = np.clip(rewards, -1.0, 1.0)

            for i, done in enumerate(np.logical_or(terminated, truncated)):
                if done:

                    if is_fetch_env:
                        terminated_log = success_step[i]
                    else:
                        terminated_log = done

                    last_episode_rewards[i] = episode_rewards[i]
                    logger.log_episode(
                        worker=i,
                        episode=worker_episodes[i],
                        total_steps=update * hp["n_steps_per_update"] + step,
                        total_reward=last_episode_rewards[i],
                        terminated=terminated_log,
                    )
                    worker_episodes[i] += 1
                    episode_rewards[i] = 0.0
                    worker_timesteps[i] = 0
                    success_step[i] = None
                    success_reached[i] = False

                    if (not hp["atari_mode"]) and i == 0:
                        episodes_finished_worker0 += 1

            rollouts["states"][step] = torch.as_tensor(states, device=device)

            if hp["is_continuous_actions"]:
                rollouts["actions"][step] = actions.detach().to(device=device, dtype=torch.float32)
            else:
                rollouts["actions"][step] = actions.detach().to(device=device, dtype=torch.long)

            rollouts["value_preds"][step] = state_value_preds.squeeze(-1).detach()
            rollouts["rewards"][step] = torch.as_tensor(rewards, device=device)
            rollouts["old_log_probs"][step] = action_log_probs.detach()
            rollouts["masks"][step] = torch.as_tensor(1.0 - np.logical_or(terminated, truncated), device=device)
            rollouts["entropies"][step] = entropy.detach()

            states = next_states

        critic_loss, actor_loss, entropy = agent.update_agent(rollouts, hp)
        critic_losses.append(critic_loss)
        actor_losses.append(actor_loss)
        entropies.append(entropy)

        logger.log_update(
            update_num=update,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            entropy=entropy,
            total_steps=total_time_steps,
            log_resources=True,
        )

        if hp["atari_mode"]:
            total_iteration += hp["n_steps_per_update"] * hp["n_envs"]
        else:
            total_iteration += episodes_finished_worker0

        episodes_finished_worker0 = 0

        mean_reward = rollouts["rewards"].sum(dim=0).mean().cpu().item()
        update_time = time.time() - update_start_time
        elapsed_time = time.time() - start_time

        if update % 5 == 0:
            print(
                f"[TRAIN] Update {update:4d} | "
                f"CriticLoss: {critic_loss:.3f} | ActorLoss: {actor_loss:.3f} | "
                f"Entropy: {entropy:.3f} | Mean last rewards: {last_episode_rewards.mean():.2f} | "
                f"Last rewards per env: {last_episode_rewards}"
            )
            print(
                f"[TIME] Update {update} | Total steps: {total_time_steps} | "
                f"Update time: {update_time:.2f}s | Elapsed: {elapsed_time/60:.2f} min"
            )

    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), f"models/a2c_episodic_agent_{hp['run_id']}.pth")
    print("[INFO] Training finished. Model saved.")

    recorder = RenderRecorder(f"video/a2c_video_{hp['run_id']}.mp4", fps=30)

    if hp["atari_mode"]:

        def make_render_env(env_id, seed=0):
            env = gym.make(env_id, frameskip=1, render_mode="rgb_array", obs_type="rgb")
            env = AtariPreprocessing(
                env,
                frame_skip=hp.get("frame_skip", 4),
                grayscale_obs=True,
                scale_obs=True,
                terminal_on_life_loss=False,
                noop_max=30,
                screen_size=84,
            )
            env = FireResetEnv(env)
            env = FrameStackObservation(env, stack_size=hp.get("stack_size", 4))
            env.reset(seed=seed)
            return env

        test_env = make_render_env(hp["env_name"], seed=hp["seed"])
    elif hp["env_name"].startswith("Fetch"):
        test_env = gym.make(hp["env_name"], max_episode_steps=50, render_mode="rgb_array")
        test_env = FetchGoalErrorWrapper(test_env)
    elif hp["env_name"] == "Pinball-v0":
        import envs.pinball

        test_env = gym.make(hp["env_name"], config_name="hard", render_mode="rgb_array")
    else:
        test_env = gym.make(hp["env_name"], render_mode="rgb_array")

    state, _ = test_env.reset(seed=hp["seed"])
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            if hp["atari_mode"]:
                state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
                action, _, _, _ = agent.select_action(state_tensor)

                if isinstance(action, torch.Tensor):
                    action_np = action.cpu().numpy()
                else:
                    action_np = action
                actions_to_env = int(np.asarray(action_np).item())
            else:
                if is_continuous:
                    actions, action_log_probs, state_value_preds, entropy, raw_actions = agent.select_action(states)
                else:
                    actions, action_log_probs, state_value_preds, entropy, _ = agent.select_action(states)
                    raw_actions = None

            if hp["is_continuous_actions"]:
                actions_to_env = actions.detach().cpu().numpy()
                if actions_to_env.ndim > 1:
                    actions_to_env = actions_to_env[0]
            else:
                actions_to_env = int(actions.detach().cpu().numpy().flatten()[0])

        next_state, reward, terminated, truncated, _ = test_env.step(actions_to_env)
        total_reward += reward

        frame = test_env.render()
        recorder.capture(frame)

        state = next_state
        done = terminated or truncated

    recorder.save()
    print(f"[RESULT] Evaluation video saved to {recorder.filename} | Total reward: {total_reward:.2f}")
