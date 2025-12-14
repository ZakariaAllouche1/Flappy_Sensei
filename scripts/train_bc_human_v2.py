import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from train_bc_v0 import BCNet

DATA_HUMAN_V1 = "data/human_demos.npz"
DATA_HUMAN_V2 = "data/human_demos_V2.npz"  
MODEL_PATH    = "models/bc_student_human_v2.pth"

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4


def load_npz(path):
    d = np.load(path)
    obs = d["obs"].astype(np.float32) / 255.0    
    obs = np.transpose(obs, (0, 3, 1, 2))        
    act = d["actions"].astype(np.int64)          
    return obs, act


def main():
    os.makedirs("models", exist_ok=True)

    obs1, act1 = load_npz(DATA_HUMAN_V1)
    obs2, act2 = load_npz(DATA_HUMAN_V2)

    print("Human V1 samples:", obs1.shape[0])
    print("Human V2 samples:", obs2.shape[0])

    # concat v1 + v2
    obs = np.concatenate([obs1, obs2], axis=0)
    act = np.concatenate([act1, act2], axis=0)

    print("Total samples:", obs.shape[0])
    print("Proportion flap (1):", (act == 1).mean())

    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(act)

    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BCNet(n_actions=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_obs, batch_act in loader:
            batch_obs = batch_obs.to(device)
            batch_act = batch_act.to(device)

            logits = model(batch_obs)
            loss = F.cross_entropy(logits, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_obs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"[BC human v2] Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("BC human v2 saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
