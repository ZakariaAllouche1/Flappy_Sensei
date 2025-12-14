import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from train_bc_v0 import BCNet  

DATA_PATH = "data/human_demos.npz"
MODEL_PATH = "models/bc_student_human.pth"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4


def main():
    os.makedirs("models", exist_ok=True)

    data = np.load(DATA_PATH)
    obs = data["obs"]        
    actions = data["actions"]  

    obs = obs.astype(np.float32) / 255.0
    obs = np.transpose(obs, (0, 3, 1, 2))

    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(actions)

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
        print(f"[BC human] Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("BC student (human) saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
