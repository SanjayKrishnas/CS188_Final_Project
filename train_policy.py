import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split

# --- Model Definition ---
class ImitationPolicy(nn.Module):
    def __init__(self, input_dim=21):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.position_head = nn.Linear(64, 3)
        self.rotation_head = nn.Linear(64, 3)
        self.gripper_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.backbone(x)
        pos = self.position_head(features)
        rot = self.rotation_head(features)
        grip = torch.tanh(self.gripper_head(features))  # Gripper in [-1, 1]
        return torch.cat([pos, rot, grip], dim=-1)


# --- Dataset ---
class ImitationDataset(Dataset):
    def __init__(self, obs, actions):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


# --- Data Loading ---
def load_demos(npz_path):
    flat_data = np.load(npz_path)
    demos = defaultdict(dict)
    for key in flat_data.files:
        parts = key.split('_', 2)
        if len(parts) < 3:
            continue
        demo_id = f"{parts[0]}_{parts[1]}"
        field_name = parts[2]
        demos[demo_id][field_name] = flat_data[key]
    return demos

def create_dataset(demos):
    obs_list, action_list = [], []

    for demo in demos.values():
        obs_vector = np.concatenate([
            demo["obs_robot0_eef_pos"],
            demo["obs_robot0_eef_quat"],
            demo["obs_object"]
        ], axis=-1)
        obs_list.append(obs_vector)
        action_list.append(demo["actions"])

    obs = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(action_list, axis=0)
    return obs, actions


# --- Training ---
def train(model, dataset, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Split dataset
    val_len = int(len(dataset) * config["val_split"])
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.MSELoss()

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for obs_batch, act_batch in train_loader:
            obs_batch, act_batch = obs_batch.to(device), act_batch.to(device)

            pred_actions = model(obs_batch)
            loss = loss_fn(pred_actions, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_obs, val_act in val_loader:
                val_obs, val_act = val_obs.to(device), val_act.to(device)
                val_pred = model(val_obs)
                val_loss += loss_fn(val_pred, val_act).item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), config["checkpoint_path"])
    print(f"Model saved to {config['checkpoint_path']}")


# --- Main ---
if __name__ == "__main__":
    config = {
        "epochs": 100,
        "batch_size": 128,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "val_split": 0.2,
        "checkpoint_path": "bc_model.pth",
        "obs_dim": 21,
        "act_dim": 7,
        "demo_dir": "demos"
    }

    demos = load_demos(f"{config['demo_dir']}.npz")
    obs, actions = create_dataset(demos)
    dataset = ImitationDataset(obs, actions)

    model = ImitationPolicy(input_dim=config["obs_dim"])
    train(model, dataset, config)
