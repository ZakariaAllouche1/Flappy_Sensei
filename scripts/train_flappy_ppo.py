import os

import gymnasium as gym
import flappy_bird_env 

from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

ENV_ID = "FlappyBird-v0"

MODEL_DIR = "models"
BASE_NAME = "expert_ppo_flappy_v1_framestack4_final"

FINAL_MODEL_PATH = os.path.join(MODEL_DIR, BASE_NAME)
BEST_MODEL_DIR = os.path.join(MODEL_DIR, BASE_NAME + "_best")

os.makedirs(MODEL_DIR, exist_ok=True)


def make_env(render_mode=None):
    def _init():
        env = gym.make(ENV_ID, render_mode=render_mode)
        env = ResizeObservation(env, (84, 84))
        env = GrayScaleObservation(env, keep_dim=True)
        env = Monitor(env)
        return env
    return _init


def make_vec_env(render_mode=None):
    venv = DummyVecEnv([make_env(render_mode=render_mode)])
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)  
    return venv


def main():
    train_env = make_vec_env(render_mode=None)
    eval_env = make_vec_env(render_mode=None)

    model = PPO(
        "CnnPolicy",
        train_env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="logs/ppo_flappy_v1_framestack4_final",
        device="auto",
    )

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=5.0,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,     
        log_path="logs/ppo_flappy_v1_framestack4_eval_final",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback,
    )

    model.learn(
        total_timesteps=2_000_000,  
        callback=eval_callback,
    )

    model.save(FINAL_MODEL_PATH)  

    train_env.close()
    eval_env.close()

    print("Entraînement terminé.")
    print(f"Dernier modèle : {FINAL_MODEL_PATH}.zip")
    print(f"Meilleur modèle : {os.path.join(BEST_MODEL_DIR, 'best_model.zip')}")


if __name__ == "__main__":
    main()
