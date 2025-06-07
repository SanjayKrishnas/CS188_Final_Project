'''

DO NOT MODIFY THIS FILE

'''

import numpy as np
import robosuite as suite
from dmp_policy import DMPPolicyWithPID



# create environment instance
env = suite.make(
    env_name="NutAssemblySquare", 
    robots="Panda", 
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True  
)

success_rate = 0
# reset the environment
for i in range(10):
    obs = env.reset()
    policy = DMPPolicyWithPID(obs['SquareNut_pos']) 
    print(obs['SquareNut_pos'])
    for _ in range(500):
        action = policy.get_action(obs['robot0_eef_pos'])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if reward == 1.0:
            success_rate += 1
            break

success_rate /= 10
print('success rate:', success_rate)