import numpy as np
import gymnasium as gym
import flappy_bird_env
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
import pygame

import torch
import torch.nn.functional as F
from train_bc_v0 import BCNet

ENV_ID = "FlappyBird-v0"

#MODEL_PATH = "models/bc_student_human.pth"
#MODEL_PATH = "models/bc_student_human_dagger1.pth"
# MODEL_PATH = "models/bc_student_human_v2.pth"
MODEL_PATH = "models/bc_student_human_v2_dagger1.pth"

N_EPISODES = 5 
FPS = 25         


def make_env():
    env = gym.make(ENV_ID, render_mode="human")
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env, keep_dim=True)
    return env


def select_action(model, obs, device):
    obs_norm = obs.astype("float32") / 255.0
    obs_norm = np.transpose(obs_norm, (2, 0, 1))  
    obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(device) 
    with torch.no_grad():
        logits = model(obs_t)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()
    return action


def main():
    pygame.init()
    clock = pygame.time.Clock()

    env = make_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Model:", MODEL_PATH)
    print("Episodes:", N_EPISODES)

    model = BCNet(n_actions=2).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_scores = []
    all_flap_ratios = []

    for ep in range(1, N_EPISODES + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        total_flaps = 0
        total_steps = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    if len(all_scores) > 0:
                        scores = np.array(all_scores, dtype=float)
                        print("\n==== Partial evaluation stats ====")
                        print(f"Episodes done: {len(all_scores)}")
                        print(f"Mean reward: {scores.mean():.4f}")
                        print(f"Std reward:  {scores.std():.4f}")
                        print(f"Min / Max:   {scores.min():.4f} / {scores.max():.4f}")
                    return

            action = select_action(model, obs, device)

            if action == 1:
                total_flaps += 1
            total_steps += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            clock.tick(FPS)

        flap_ratio = total_flaps / total_steps if total_steps > 0 else 0.0
        all_scores.append(total_reward)
        all_flap_ratios.append(flap_ratio)

        print(f"[Eval] Episode {ep}: reward = {total_reward:.4f}, "
              f"flap_ratio = {flap_ratio:.4f} ({total_flaps}/{total_steps})")

    env.close()
    pygame.quit()

    scores = np.array(all_scores, dtype=float)
    flap_ratios = np.array(all_flap_ratios, dtype=float)

    print("\n==== Evaluation stats ====")
    print(f"Episodes: {N_EPISODES}")
    print(f"Mean reward: {scores.mean():.4f}")
    print(f"Std reward:  {scores.std():.4f}")
    print(f"Min / Max:   {scores.min():.4f} / {scores.max():.4f}")
    print(f"Mean flap ratio: {flap_ratios.mean():.4f}")


if __name__ == "__main__":
    main()
