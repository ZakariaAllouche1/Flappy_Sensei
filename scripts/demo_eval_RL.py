import gymnasium as gym
import flappy_bird_env  

from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

ENV_ID = "FlappyBird-v0"

# Choisir ici :
ALGO = "dqn"  # "dqn" ou "ppo"
# ALGO = "ppo"
MODEL_PATH = "models/expert_dqn_flappy_v3"
# MODEL_PATH = "models/expert_ppo_flappy_v1_framestack4"

N_EPISODES = 10  


def make_env(render_mode="human"):
    """
    Crée l'env EXACTEMENT comme à l'entraînement,
    mais avec render_mode="human" pour afficher la fenêtre.
    """
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env, keep_dim=True)
        return env
    return _init


def main():
    venv = DummyVecEnv([make_env(render_mode="human")])
    venv = VecFrameStack(venv, n_stack=4)

    if ALGO.lower() == "dqn":
        model = DQN.load(MODEL_PATH, env=venv)
    elif ALGO.lower() == "ppo":
        model = PPO.load(MODEL_PATH, env=venv)
    else:
        raise ValueError("ALGO doit être 'dqn' ou 'ppo'")

    base_env = venv.venv.envs[0]

    for ep in range(N_EPISODES):
        obs = venv.reset()     
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = venv.step(action)
            total_reward += float(rewards[0])
            done = bool(dones[0])
            base_env.render()

        print(f"[{ALGO.upper()} demo] Épisode {ep+1}/{N_EPISODES} - reward = {total_reward:.3f}")

    venv.close()
    print("Demo terminée")


if __name__ == "__main__":
    main()
