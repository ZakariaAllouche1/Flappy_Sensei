import gymnasium as gym
import flappy_bird_env  

from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

ENV_ID = "FlappyBird-v0"
MODEL_PATH = "models/expert_dqn_flappy_v3"


def make_env():
    def _init():
        env = gym.make(ENV_ID)
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env, keep_dim=True)
        return env
    return _init


def main():
    venv = DummyVecEnv([make_env()])
    venv = VecFrameStack(venv, n_stack=4)

    model = DQN(
        "CnnPolicy",
        venv,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=32,
        buffer_size=100_000,          
        learning_starts=20_000,
        train_freq=4,
        target_update_interval=10_000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        max_grad_norm=10,
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=5_000_000)

    model.save(MODEL_PATH)
    venv.close()
    print(f"Expert DQN v3 saved to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
