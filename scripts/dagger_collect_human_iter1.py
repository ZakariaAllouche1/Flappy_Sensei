import os
import numpy as np
import gymnasium as gym
import flappy_bird_env
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation
import pygame

import torch
import torch.nn.functional as F
from train_bc_v0 import BCNet 

ENV_ID = "FlappyBird-v0"

# BC_MODEL_PATH = "models/bc_student_human.pth"
# OUT_PATH = "data/dagger_human_iter1.npz"

BC_MODEL_PATH = "models/bc_student_human_v2.pth"
OUT_PATH = "data/dagger_human_iter1_v2.npz"

N_EPISODES = 10
FPS = 25  


def make_env():
    env = gym.make(ENV_ID, render_mode="human")
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env, keep_dim=True)
    return env


def select_action_bc(model, obs, device):
    obs_norm = obs.astype("float32") / 255.0
    obs_norm = np.transpose(obs_norm, (2, 0, 1)) 
    obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(device) 
    with torch.no_grad():
        logits = model(obs_t)
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()
    return action


def main():
    os.makedirs("data", exist_ok=True)

    pygame.init()
    clock = pygame.time.Clock()

    env = make_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    bc_model = BCNet(n_actions=2).to(device)
    state_dict = torch.load(BC_MODEL_PATH, map_location=device)
    bc_model.load_state_dict(state_dict)
    bc_model.eval()

    dagger_obs = []
    dagger_actions = []
    scores = []

    running = True

    for ep in range(N_EPISODES):
        if not running:
            break

        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done and running:
            human_flap = None 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                        human_flap = 1  
                    elif event.key == pygame.K_DOWN:
                        human_flap = 0  

            a_bc = select_action_bc(bc_model, obs, device)

            if human_flap is None:
                a_expert = a_bc
            else:
                a_expert = human_flap

            dagger_obs.append(obs)
            dagger_actions.append(int(a_expert))

            env_action = a_expert

            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated
            total_reward += reward

            clock.tick(FPS)

        if not running:
            break

        scores.append(total_reward)
        print(f"[DAgger human] Episode {ep+1}/{N_EPISODES}: reward = {total_reward}")

    env.close()
    pygame.quit()

    if len(dagger_obs) == 0:
        print("Aucune donnée DAgger enregistrée.")
        return

    observations = np.array(dagger_obs, dtype=np.uint8)   
    actions = np.array(dagger_actions, dtype=np.int64)    

    np.savez(OUT_PATH, obs=observations, actions=actions)
    print("DAgger human data saved to", OUT_PATH)

    if len(scores) > 0:
        scores = np.array(scores, dtype=float)
        print("==== DAgger human rollout stats ====")
        print(f"Mean reward: {scores.mean():.4f} ± {scores.std():.4f}")


if __name__ == "__main__":
    main()
