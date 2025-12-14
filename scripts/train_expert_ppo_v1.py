import gymnasium as gym
import flappy_bird_env

from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

ENV_ID = "FlappyBird-v0"
MODEL_PATH = "models/expert_ppo_flappy_v1_framestack4"


def make_env():
    def _init():
        env = gym.make(ENV_ID)
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env, keep_dim=True)
        return env
    return _init


def main():
    # VecEnv + FrameStack
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="auto",  # GPU si dispo
    )

    model.learn(total_timesteps=1_000_000)

    model.save(MODEL_PATH)
    env.close()
    print(f"Expert v1 saved to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
