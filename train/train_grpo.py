import time
import os
import random
import torch
import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from A2C import A2C, EpisodeBuffer
from util.Logger import Logger
from util.RenderRecorder import RenderRecorder
from util.AtariUtils import FireResetEnv


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []

    def add(self, episode):
        self.buffer.append(episode)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_episodes, hist_frac=0.1):
        if len(self.buffer) == 0:
            return []

        if len(self.buffer) <= batch_episodes:
            return self.buffer.copy()

        n_hist = int(batch_episodes * hist_frac)
        n_recent = batch_episodes - n_hist

        recent_pool = self.buffer[-batch_episodes:]
        hist_pool = self.buffer[:-batch_episodes]

        recent_sel = recent_pool[:n_recent]
        hist_sel = random.sample(hist_pool, min(n_hist, len(hist_pool))) if n_hist > 0 and len(hist_pool) > 0 else []

        return recent_sel + hist_sel


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
            else:
                return gym.make(env_name)

        return _init

    return SyncVectorEnv([make_single_env() for _ in range(num_envs)])


def record_render(hp, agent, device):
    if RenderRecorder is not None:
        recorder = RenderRecorder(f"grpo_video_{hp['run_id']}.mp4", fps=30)
        if hp["atari_mode"]:

            def make_render_env(env_id, seed=0):
                env = gym.make(env_id, frameskip=1, render_mode="rgb_array", obs_type="rgb")
                env = AtariPreprocessing(
                    env,
                    frame_skip=hp["frame_skip"],
                    grayscale_obs=True,
                    scale_obs=True,
                    terminal_on_life_loss=False,
                    noop_max=30,
                    screen_size=84,
                )
                env = FireResetEnv(env)
                env = FrameStackObservation(env, stack_size=hp["stack_size"])
                env.reset(seed=seed)
                return env

            test_env = make_render_env(hp["env_name"], seed=hp["seed"])
        else:
            if hp["env_name"] == "Pinball-v0":
                import envs.pinball

                test_env = gym.make(hp["env_name"], config_name="hard", render_mode="rgb_array")
            else:
                test_env = gym.make(hp["env_name"], render_mode="rgb_array")
            test_env.reset(seed=hp["seed"])
        state, _ = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _, _ = agent.select_action(s)
                if hp["is_continuous_actions"]:
                    action_to_env = action.cpu().numpy()[0]
                else:
                    action_to_env = int(action.item())
            next_state, reward, terminated, truncated, _ = test_env.step(action_to_env)
            total_reward += reward
            frame = test_env.render()
            recorder.capture(frame)
            state = next_state
            done = terminated or truncated
        recorder.save()
        print(f"[RESULT] Video saved to {recorder.filename} | Reward {total_reward:.2f}")


def train_grpo(hp, device):

    if hp["atari_mode"]:
        gym.register_envs(ale_py)
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

    if hp["is_continuous_actions"]:
        act_space = envs.single_action_space.shape[0]
    else:
        act_space = envs.single_action_space.n

    agent = A2C(
        agent_type=hp["agent_type"],
        n_features=obs_space,
        n_actions=act_space,
        device=device,
        critic_lr=hp["critic_lr"],
        actor_lr=hp["actor_lr"],
        n_envs=hp["n_envs"],
        atari_mode=hp["atari_mode"],
        ppo_epochs=hp["ppo_epochs"],
        ppo_batch_size=hp["ppo_batch_size"],
        clip_coef=hp["clip_coef"],
        adv_clip=None,
        stack_size=hp["stack_size"],
        use_kl=hp["use_kl"],
        is_continuous_actions=hp["is_continuous_actions"],
    )

    logger = Logger(
        episode_filename=f"logs/grpo_episodes_{hp['run_id']}.csv",
        update_filename=f"logs/grpo_updates_{hp['run_id']}.csv",
        resources_filename=f"logs/grpo_resources_{hp['run_id']}.csv",
    )

    episode_buffer = EpisodeBuffer(n_envs=hp["n_envs"], device=device)
    replay = ReplayBuffer(max_size=hp["replay_max_size"])

    states, _ = envs.reset(seed=hp["seed"])
    episode_rewards = np.zeros(hp["n_envs"], dtype=np.float32)
    last_episode_rewards = np.zeros(hp["n_envs"], dtype=np.float32)
    worker_episodes = np.zeros(hp["n_envs"], dtype=int)
    worker_timesteps = np.zeros(hp["n_envs"], dtype=int)

    total_time_steps = 0
    update = 0
    start_time = time.time()
    replay_buffer_counter = 0

    if hp["atari_mode"]:
        max_iterations = hp["max_episode_steps"]
    else:
        max_iterations = hp["max_episodes"]

    batch_episodes_size = hp["grpo_batch_episodes"]
    replay_frac = hp["replay_frac"]

    while worker_episodes[0] < max_iterations:
        update_start_time = time.time()
        worker_timesteps += 1

        with torch.no_grad():
            state_tensor = torch.as_tensor(states, dtype=torch.float32, device=device)
            actions, action_log_probs, logits, entropy = agent.select_action(state_tensor)
            pre_tanh = getattr(agent, "_pre_tanh_action", None)
            old_std = torch.exp(agent.log_std).detach() if hp["is_continuous_actions"] else None

        if hp["is_continuous_actions"]:
            env_action = actions.cpu().numpy()
        else:
            env_action = actions.cpu().numpy()

        next_states, rewards, terminated, truncated, infos = envs.step(env_action)

        total_time_steps += hp["n_envs"]
        episode_rewards += rewards

        clipped_rewards = np.clip(rewards, -1.0, 1.0) if hp["atari_mode"] else rewards

        for i in range(hp["n_envs"]):

            if hp["is_continuous_actions"]:
                a = actions[i]
                lp = action_log_probs[i]
                pt = pre_tanh[i] if pre_tanh is not None else None
                os_ = old_std.expand_as(a)
            else:
                a = actions[i].unsqueeze(0)
                lp = action_log_probs[i].unsqueeze(0)
                pt = None
                os_ = None

            episode_buffer.add_step(
                i,
                states[i],
                a,
                lp,
                logits[i],
                torch.tensor(clipped_rewards[i], device=device),
                pre_tanh_action=pt,
                old_std=os_,
            )

        done_mask = np.logical_or(terminated, truncated)
        for i, done in enumerate(done_mask):
            if done:
                logger.log_episode(
                    worker=i,
                    episode=worker_episodes[i],
                    total_steps=worker_timesteps[i],
                    total_reward=episode_rewards[i],
                    terminated=done,
                )
                last_episode_rewards[i] = episode_rewards[i]
                worker_episodes[i] += 1
                episode_rewards[i] = 0
                worker_timesteps[i] = 0

                ep = episode_buffer.end_episode(i)
                if ep is not None:
                    replay.add(ep)
                    replay_buffer_counter += 1

        states = next_states

        if replay_buffer_counter >= batch_episodes_size:
            update += 1
            replay_buffer_counter = 0

            to_update = replay.sample_batch(batch_episodes_size, replay_frac)

            critic_loss, actor_loss, ent = agent.update_agent(to_update, hp)

            logger.log_update(
                update_num=update,
                critic_loss=critic_loss,
                actor_loss=actor_loss,
                entropy=ent,
                total_steps=total_time_steps,
                log_resources=True,
            )

            update_time = time.time() - update_start_time
            elapsed = time.time() - start_time

            print(
                f"[TRAIN] Update {update} | ActorLoss {actor_loss:.4f} | Entropy {ent:.4f} | Reward {last_episode_rewards.mean():.2f}"
            )
            print(f"[TIME] Steps {total_time_steps} | Update {update_time:.2f}s | Elapsed {elapsed/60:.2f}m")

            # record_render(hp, agent, device)

    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), f"models/grpo_episodic_agent_{hp['run_id']}.pth")
    print("[INFO] Training finished. Model saved.")
    record_render(hp, agent, device)
