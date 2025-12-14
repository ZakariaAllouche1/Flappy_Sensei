import numpy as np
import gymnasium as gym
import flappy_bird_env
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
import pygame
import os

ENV_ID = "FlappyBird-v0"
# OUT_PATH = "data/human_demos.npz"
OUT_PATH = "data/human_demos_V2.npz"
N_EPISODES = 20   

FPS = 25


def make_env():
    env = gym.make(ENV_ID, render_mode="human")
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env, keep_dim=True)
    return env


def main():
    os.makedirs("data", exist_ok=True)

    pygame.init()
    clock = pygame.time.Clock()

    env = make_env()

    all_obs = []
    all_actions = []
    scores = []

    running = True

    for ep in range(N_EPISODES):
        if not running:
            break

        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done and running:
            flap = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                        flap = 1

            action = flap 

            all_obs.append(obs)
            all_actions.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            clock.tick(FPS)

        if not running:
            break

        scores.append(total_reward)
        print(f"[Human demos] Episode {ep+1}/{N_EPISODES}: reward = {total_reward}")

    env.close()
    pygame.quit()

    if len(all_obs) == 0:
        print("Aucune donnée enregistrée (fenêtre fermée trop tôt).")
        return

    observations = np.array(all_obs, dtype=np.uint8)  
    actions = np.array(all_actions, dtype=np.int64)  

    np.savez(OUT_PATH, obs=observations, actions=actions)
    print("Human demos saved to", OUT_PATH)

    if len(scores) > 0:
        scores = np.array(scores, dtype=float)
        print("==== Human performance during recording ====")
        print(f"Mean reward: {scores.mean():.4f} ± {scores.std():.4f}")


if __name__ == "__main__":
    main()
