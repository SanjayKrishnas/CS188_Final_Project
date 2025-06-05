import numpy as np
import torch
from train_policy import ImitationPolicy  # assumes both scripts are in the same dir
import robosuite as suite
from robosuite.utils.placement_samplers import UniformRandomSampler

# Load trained model
policy = ImitationPolicy(input_dim=21)
policy.load_state_dict(torch.load("bc_model.pth"))
policy.eval()

def get_action(obs):
    # Construct input: object + eef pos + eef quat + gripper qpos
    input_vec = np.concatenate([
        obs["robot0_eef_pos"],
        obs["robot0_eef_quat"],
        obs["object-state"]
    ])
    with torch.no_grad():
        action = policy(torch.tensor(input_vec, dtype=torch.float32)).numpy()
    return action

placement_initializer = UniformRandomSampler(
    name="RandomOriSampler",
    mujoco_objects=None,
    x_range=[-0.115, -0.11],
    y_range=[0.05, 0.225],
    rotation=[np.pi],
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
    control_freq=20, 
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True, 
    placement_initializer=placement_initializer,
    ignore_done=True,
)

success_rate = 0
for _ in range(5):
    obs = env.reset()
    for _ in range(2500):
        action = get_action(obs)
        for _ in range (10):
            obs, reward, done, info = env.step(action)
            env.render()
            if reward == 1.0:
                success_rate += 1
                break

print("Success rate:", success_rate / 5.0)


# import numpy as np
# import itertools
# import robosuite as suite

# # Create environment
# env = suite.make(
#     env_name="NutAssemblySquare",
#     robots="Panda",
#     has_renderer=False,
#     use_camera_obs=False,
#     use_object_obs=True
# )

# # Reset to get observation
# obs = env.reset()

# # Extract object-state
# object_state = obs["object-state"]
# print("object-state shape:", object_state.shape)

# # Collect candidate keys and their shapes
# candidates = [(k, v.shape[0]) for k, v in obs.items() if isinstance(v, np.ndarray) and len(v.shape) == 1]

# # Try combinations of 2 to 4 keys whose total dim is 14
# for r in range(2, 5):
#     for combo in itertools.combinations(candidates, r):
#         keys, dims = zip(*combo)
#         if sum(dims) == 14:
#             concatenated = np.concatenate([obs[k] for k in keys])
#             if np.allclose(concatenated, object_state, atol=1e-6):
#                 print(f"\nMatch found! 'object-state' == [{', '.join(keys)}]")


