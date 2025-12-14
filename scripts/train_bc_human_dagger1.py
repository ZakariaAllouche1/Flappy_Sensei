import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from train_bc_v0 import BCNet

DATA_FILES = [
    "data/human_demos.npz",          # Human V1
    "data/human_demos_V2.npz",       # Human V2
    "data/dagger_human_iter1_v2.npz" # DAgger sur BC_v2
]

BASE_MODEL_PATH = "models/bc_student_human_v2.pth"         
OUT_MODEL_PATH  = "models/bc_student_human_v2_dagger1.pth"  

BATCH_SIZE = 64
EPOCHS = 5           # fine tuning leger
LR = 1e-4


def load_and_concat(files):
    all_obs = []
    all_act = []
    for path in files:
        d = np.load(path)
        all_obs.append(d["obs"])
        all_act.append(d["actions"])
        print(f"Loaded {path}: {d['obs'].shape[0]} samples")
    obs = np.concatenate(all_obs, axis=0)
    act = np.concatenate(all_act, axis=0)
    return obs, act


def main():
    os.makedirs("models", exist_ok=True)

    obs, actions = load_and_concat(DATA_FILES)

    print("Total dataset size:", obs.shape[0])
    print("Proportion flap (1):", (actions == 1).mean())

    obs = obs.astype(np.float32) / 255.0
    obs = np.transpose(obs, (0, 3, 1, 2))
    actions = actions.astype(np.int64)

    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(actions)

    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BCNet(n_actions=2).to(device)
    state_dict = torch.load(BASE_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    class_weights = torch.tensor([1.0, 2.0], device=device)  # [w_action0, w_action1]

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            logits = model(batch_obs)
            loss = F.cross_entropy(logits, batch_act, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_obs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[BC_v2 + DAgger1] Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), OUT_MODEL_PATH)
    print("BC_v2 + DAgger1 saved to", OUT_MODEL_PATH)


if __name__ == "__main__":
    main()
