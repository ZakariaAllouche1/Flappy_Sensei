import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = "data/expert_demos_v0.npz"
MODEL_PATH = "models/bc_student_v0.pth"
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3


class BCNet(nn.Module):
    """
    CNN:
    entrée : (B, 1, 84, 84)
    sortie : logits sur 2 actions (0 / 1)
    """
    def __init__(self, n_actions: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # calcul de la taille du flatten
        with torch.no_grad():
            x = torch.zeros(1, 1, 84, 84)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            flat_dim = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 512)
        self.fc_out = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: (B, 1, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        return self.fc_out(x)


def main():
    os.makedirs("models", exist_ok=True)

    data = np.load(DATA_PATH)
    obs = data["obs"]         
    actions = data["actions"] 

    # normalisation 
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
        print(f"Epoch {epoch}/{EPOCHS} - loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("BC student saved to", MODEL_PATH)


if __name__ == "__main__":
    main()
