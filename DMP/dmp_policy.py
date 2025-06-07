import numpy as np
from collections import defaultdict
from dmp import DMP
from pid import PID
from load_data import reconstruct_from_npz

class DMPPolicyWithPID:
    """
    A policy that follows a demonstrated path with DMPs and PID control.

    The demonstration is split into segments based on grasp toggles.  
    The first segment's endpoint is re-targeted to a new object pose.
    Subsequent segments replay the original DMP rollouts.

    Args:
        square_obs (dict): 'SquareNut_pos' observed
        demo_path (str): path to .npz file with demo data.
        dt (float): control timestep.
        n_bfs (int): number of basis functions per DMP.
    """
    def __init__(self, square_pos, demo_path='../demos.npz', dt=0.01, n_bfs=20):
        self.dt = dt
        self.n_bfs = n_bfs
        
        # Load and parse best demo
        demos = reconstruct_from_npz(demo_path)
        demo = demos['demo_0']
        best_error = 1000
        selected_demo_idx = 0
        
        # Use your original min X-coordinate method (since it works better)
        for i in range(200):
            demo_candidate = demos[f'demo_{i}']
            positions = demo_candidate['obs_robot0_eef_pos']
            demo_nut_pos = min(positions, key=lambda x: x[0])  # Your original method
            error = np.linalg.norm(demo_nut_pos - square_pos)
            if error < best_error:
                best_error = error
                demo = demo_candidate
                selected_demo_idx = i
        
        print(f"Selected demo {selected_demo_idx} with min-X method error: {best_error:.3f}")

        # Extract trajectories and grasp
        ee_pos = demo['obs_robot0_eef_pos']  # (T,3)
        ee_grasp = demo['actions'][:, -1:].astype(int)  # (T,1)
        segments = self.detect_grasp_segments(ee_grasp)

        # Compute offset for first segment to new object pose
        demo_obj_pos = demo['obs_object'][0, :3]
        new_obj_pos = square_pos
        start, end = segments[0]
        offset = ee_pos[end-1] - demo_obj_pos

        self.segments = segments
        self.segment_grasp_states = []
        self.dmps = []
        self.segment_trajectories = []
        
        for i, (start, end) in enumerate(segments):
            segment_traj = ee_pos[start:end] 
            grasp_state = ee_grasp[start, 0]
            self.segment_grasp_states.append(grasp_state)
            
            dmp = DMP(n_dmps=3, n_bfs=n_bfs, dt=dt,
            y0=segment_traj[0],
            goal=segment_traj[-1])
            dmp.imitate(segment_traj)
            self.dmps.append(dmp)
            
            if i == 0:
                # First segment: approach and pickup
                new_goal = new_obj_pos + offset
                
                demo_nut_pos = min(demo['obs_robot0_eef_pos'], key=lambda x: x[0])
                approach_from_right = demo_nut_pos[1] > 0.17
                
                if approach_from_right:
                    # When picking up from right, go slightly lower to avoid grazing
                    new_goal[2] -= 0.04  # Move 2cm DOWN for better contact
                    print(f"Right pickup approach - adjusting goal DOWN by 2cm for better contact")
                    print(f"Adjusted pickup goal: {new_goal}")
                
                traj = dmp.rollout(new_goal=new_goal)
            elif i == len(segments) - 2:
                original_goal = segment_traj[-1]  # End of this segment in demo
                
                # Determine approach direction using absolute threshold
                demo_nut_pos = min(demo['obs_robot0_eef_pos'], key=lambda x: x[0])
                approach_from_right = demo_nut_pos[1] > 0.17  # Use your determined threshold
                
                # Apply correction based on approach direction
                corrected_goal = original_goal.copy()
                if approach_from_right:
                    # Robot approaches from right:
                    # - Overshoots right -> move goal left (Y-axis)
                    # - Undershoots forward -> move goal more forward (X-axis)
                    # - Make it go HIGHER (Z-axis)
                    corrected_goal[1] -= 0.04  # (Y-axis)
                    corrected_goal[0] += 0.02  # (X-axis)
                    corrected_goal[2] += 0.03  # (Z-axis)
                    print(f"Right approach detected (Y={demo_nut_pos[1]:.3f} > 0.15)")
                    print(f"  Moving goal left by 3cm (Y), forward by 3cm (X), and UP by 5cm (Z)")
                else:
                    # Robot approaches from left:
                    # - Overshoots left -> move goal right (Y-axis)  
                    # - Overshoots forward -> move goal more backward (X-axis)
                    # - Make it go HIGHER (Z-axis)
                    corrected_goal[1] += 0.05  # (Y-axis)
                    corrected_goal[0] -= 0.06  # (X-axis)
                    corrected_goal[2] += 0.03  # (Z-axis)
                    print(f"Left approach detected (Y={demo_nut_pos[1]:.3f} <= 0.15)")
                    print(f"  Moving goal right by 3cm (Y), backward by 3cm (X), and UP by 5cm (Z)")
                
                traj = dmp.rollout(new_goal=corrected_goal)
                print(f"Original goal: {original_goal}")
                print(f"Corrected goal: {corrected_goal}")
                print(f"Corrections: ΔY={corrected_goal[1]-original_goal[1]:.3f}, ΔX={corrected_goal[0]-original_goal[0]:.3f}, ΔZ={corrected_goal[2]-original_goal[2]:.3f}")
            else:
                traj = dmp.rollout()
            
            self.segment_trajectories.append(traj)
        
        initial_target = np.zeros(3) 
        self.pid = PID(kp=[10.0, 10.0, 10.0], ki=[0.1, 0.1, 0.1], kd=[1.0, 1.0, 1.0], target=initial_target)
        self.current_segment = 0
        self.current_step = 0
        self.total_steps_executed = 0
        self.segment_lengths = [len(traj) for traj in self.segment_trajectories]
        self.cumulative_lengths = np.cumsum([0] + self.segment_lengths)

    def detect_grasp_segments(self, grasp_flags: np.ndarray) -> list:
        """
        Identify segments based on grasp toggles.

        Args:
            grasp_flags (np.ndarray): (T,1) array of grasp signals.

        Returns:
            List[Tuple[int,int]]: start and end indices per segment.
        """
        segments = []
        grasp_flat = grasp_flags.flatten()
        
        transitions = []
        for i in range(1, len(grasp_flat)):
            if grasp_flat[i] != grasp_flat[i-1]:
                transitions.append(i)
        
        start_idx = 0
        for transition_idx in transitions:
            segments.append((start_idx, transition_idx))
            start_idx = transition_idx
        
        if start_idx < len(grasp_flat):
            segments.append((start_idx, len(grasp_flat)))
        
        segments = [(start, end) for start, end in segments if end > start]
        
        return segments

    def get_action(self, robot_eef_pos: np.ndarray) -> np.ndarray:
        """
        Compute next action for the robot's end-effector.

        Args:
            robot_eef_pos (np.ndarray): Current end-effector position [x,y,z].

        Returns:
            np.ndarray: Action vector [dx,dy,dz,0,0,0,grasp].
        """
        if self.current_segment >= len(self.segments):
            return np.zeros(7)
        
        current_traj = self.segment_trajectories[self.current_segment]
        
        if self.current_step >= len(current_traj):
            self.current_segment += 1
            self.current_step = 0
            self.pid.reset()
            
            if self.current_segment >= len(self.segments):
                return np.zeros(7)
            
            current_traj = self.segment_trajectories[self.current_segment]
        
        target_pos = current_traj[self.current_step]
        
        self.pid.target = target_pos
        delta_pos = self.pid.update(robot_eef_pos, dt=self.dt)
        grasp_state = self.segment_grasp_states[self.current_segment]
        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0.0, 0.0, 0.0, float(grasp_state)])
        self.current_step += 1
        self.total_steps_executed += 1
        
        return action