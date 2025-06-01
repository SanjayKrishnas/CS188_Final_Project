import numpy as np
import torch
from train_policy import ImitationPolicy  # assumes both scripts are in the same dir
import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler

# Load trained model
policy = ImitationPolicy(input_dim=23, output_dim=7)
policy.load_state_dict(torch.load("imitation_policy.pt"))
policy.eval()

def get_action(obs):
    # Construct input: object + eef pos + eef quat + gripper qpos
    input_vec = np.concatenate([
        obs["obs_object"],
        obs["obs_robot0_eef_pos"],
        obs["obs_robot0_eef_quat"],
        obs["obs_robot0_gripper_qpos"]
    ])
    with torch.no_grad():
        action = policy(torch.tensor(input_vec, dtype=torch.float32)).numpy()
    return action

placement_initializer = UniformRandomSampler(
    name="RandomOriSampler",
    mujoco_objects=None,
    x_range=[-0.115, -0.11],
    y_range=[0.05, 0.225],
    rotation=[0, 2 * np.pi],
    rotation_axis="z",
    ensure_object_boundary_in_range=False,
    ensure_valid_placement=False,
    reference_pos=(0, 0, 0.82),
    z_offset=0.02,
)

env = suite.make(
    env_name="NutAssemblySquare",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    placement_initializer=placement_initializer,
    ignore_done=True,
)

success_rate = 0
for _ in range(5):
    obs = env.reset()
    for _ in range(2500):
        action = get_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if reward == 1.0:
            success_rate += 1
            break

print("Success rate:", success_rate / 5.0)
