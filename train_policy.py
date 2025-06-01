import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

# Define MLP policy
class ImitationPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def load_demos(npz_path):
    flat_data = np.load(npz_path)
    demos = defaultdict(lambda: {})
    for key in flat_data.files:
        parts = key.split('_', 2)
        if len(parts) < 3:
            continue
        demo_id = f"{parts[0]}_{parts[1]}"
        field_name = parts[2]
        demos[demo_id][field_name] = flat_data[key]
    return demos

def create_dataset(demos):
    obs_list = []
    action_list = []
    for demo in demos.values():
        obs = np.concatenate([
            demo["obs_object"],
            demo["obs_robot0_eef_pos"],
            demo["obs_robot0_eef_quat"],
            demo["obs_robot0_gripper_qpos"]
        ], axis=-1)
        actions = demo["actions"]
        obs_list.append(obs)
        action_list.append(actions)
    return np.concatenate(obs_list, axis=0), np.concatenate(action_list, axis=0)

def train(model, obs, actions, epochs=50, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        perm = torch.randperm(obs.size(0))
        obs, actions = obs[perm], actions[perm]
        for i in range(0, obs.size(0), batch_size):
            o_batch = obs[i:i+batch_size]
            a_batch = actions[i:i+batch_size]
            pred = model(o_batch)
            loss = loss_fn(pred, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "imitation_policy.pt")
    print("Policy saved to imitation_policy.pt")

if __name__ == "__main__":
    demos = load_demos("demos.npz")
    obs, actions = create_dataset(demos)
    model = ImitationPolicy(input_dim=23, output_dim=7)  # Adjust input_dim if you change features
    train(model, obs, actions)
