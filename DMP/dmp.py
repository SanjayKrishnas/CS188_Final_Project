import numpy as np

class CanonicalSystem:
    """
    Skeleton of the discrete canonical dynamical system.
    """
    def __init__(self, dt: float, ax: float = 1.0):
        """
        Args:
            dt (float): Timestep duration.
            ax (float): Gain on the canonical decay.
        """
        # Initialize time parameters
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = 1.0
        self.timesteps: int = int(self.run_time / dt)
        self.x: float = 1.0  # phase variable

    def reset(self) -> None:
        """
        Reset the phase variable to its initial value.
        """
        self.x = 1.0

    def __step_once(self, x: float, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        dx = -self.ax * x * self.dt / tau
        return x + dx

    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Advance the phase by one timestep.

        Returns:
            float: Updated phase value.
        """
        self.x = self.__step_once(self.x, tau, error_coupling)
        return self.x

    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        """
        Generate the entire phase sequence.

        Returns:
            np.ndarray: Array of phase values over time.
        """
        xs = np.zeros(self.timesteps)
        self.reset()
        for i in range(self.timesteps):
            xs[i] = self.step(tau, ec)
        return xs

class DMP:
    """
    Skeleton of the discrete Dynamic Motor Primitive.
    """
    def __init__(
        self,
        n_dmps: int,
        n_bfs: int,
        dt: float = 0.01,
        y0: float = 0.0,
        goal: float = 1.0,
        ay: float = 25.0,
        by: float = None
    ):
        """
        Args:
            n_dmps (int): Number of dimensions.
            n_bfs (int): Number of basis functions per dimension.
            dt (float): Timestep duration.
            y0 (float|array): Initial state.
            goal (float|array): Goal state.
            ay (float|array): Attractor gain.
            by (float|array): Damping gain.
        """
        self.n_dmps: int = n_dmps
        self.n_bfs: int = n_bfs
        self.dt: float = dt
        
        self.y0: np.ndarray = np.ones(n_dmps) * y0 if np.isscalar(y0) else np.array(y0)
        self.goal: np.ndarray = np.ones(n_dmps) * goal if np.isscalar(goal) else np.array(goal)
        self.ay: np.ndarray = np.ones(n_dmps) * ay if np.isscalar(ay) else np.array(ay)
        
        if by is None:
            self.by: np.ndarray = self.ay / 4.0
        else:
            self.by: np.ndarray = np.ones(n_dmps) * by if np.isscalar(by) else np.array(by)
        
        self.w: np.ndarray = np.zeros((n_dmps, n_bfs))
        self.cs: CanonicalSystem = CanonicalSystem(dt=dt)
        
        self.c = np.exp(-self.cs.ax * np.linspace(0, self.cs.run_time, n_bfs))
        self.h = np.ones(n_bfs) * n_bfs**1.5 / self.c / self.cs.ax
        
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset trajectories and canonical system state.
        """
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset()

    def basis_functions(self, x: float) -> np.ndarray:
        psi = np.exp(-self.h * (x - self.c)**2)
        return psi / np.sum(psi)  

    def forcing_term(self, x: float, dmp_idx: int = 0) -> float:
        psi = self.basis_functions(x)
        return x * (self.goal[dmp_idx] - self.y0[dmp_idx]) * np.dot(psi, self.w[dmp_idx, :])

    def imitate(self, y_des: np.ndarray) -> np.ndarray:
        """
        Learn DMP weights from a demonstration.

        Args:
            y_des (np.ndarray): Desired trajectory, shape (T, D) or (T,) for 1D.

        Returns:
            np.ndarray: Interpolated demonstration (T', D).
        """
        demonstration = y_des.reshape(-1, 1) if y_des.ndim == 1 else y_des
        original_length, dimensions = demonstration.shape
        
        time_original = np.linspace(0, self.cs.run_time, original_length)
        time_target = np.linspace(0, self.cs.run_time, self.cs.timesteps)
        
        interpolated = np.array([np.interp(time_target, time_original, demonstration[:, i]) 
                            for i in range(min(dimensions, self.n_dmps))]).T
        
        derivatives = [interpolated]
        for order in range(2):
            derivatives.append(np.gradient(derivatives[-1], axis=0) / self.dt)
        
        positions, velocities, accelerations = derivatives
        canonical_phase = self.cs.rollout()
        
        weight_matrix = np.zeros((self.n_dmps, self.n_bfs))
        
        for axis in range(self.n_dmps):
            target_force = accelerations[:, axis] - self.ay[axis] * (
                self.by[axis] * (self.goal[axis] - positions[:, axis]) - velocities[:, axis])
            
            basis_activations = np.array([self.basis_functions(phase) for phase in canonical_phase])
            weighted_basis = basis_activations * (canonical_phase[:, None] * 
                                                (self.goal[axis] - self.y0[axis]))
            
            weight_matrix[axis] = np.linalg.lstsq(weighted_basis, target_force, rcond=None)[0]
        
        self.w = weight_matrix
        return interpolated.T if self.n_dmps == 1 else interpolated

    def rollout(
        self,
        tau: float = 1.0,
        error: float = 0.0,
        new_goal: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a new trajectory from the DMP.

        Args:
            tau (float): Temporal scaling.
            error (float): Feedback coupling.
            new_goal (np.ndarray, optional): Override goal.

        Returns:
            np.ndarray: Generated trajectory (T x D).
        """
        self.reset_state()
   
        target = np.ones(self.n_dmps) * new_goal if np.isscalar(new_goal) else (new_goal if new_goal is not None else self.goal)
        
        trajectory_data = {
            'position': np.zeros((self.cs.timesteps, self.n_dmps)),
            'velocity': np.zeros((self.cs.timesteps, self.n_dmps)),
            'acceleration': np.zeros((self.cs.timesteps, self.n_dmps))
        }
        
        state_history = []
        
        while len(state_history) < self.cs.timesteps:
            current_phase = self.cs.x
            step_idx = len(state_history)
            
            forcing_vector = np.array([current_phase * (target[dim] - self.y0[dim]) * 
                                        np.dot(self.basis_functions(current_phase), self.w[dim, :]) 
                                        for dim in range(self.n_dmps)])
            
            dynamics_term = self.ay * (self.by * (target - self.y) - self.dy * tau)
            self.ddy = (dynamics_term + forcing_vector) / (tau ** 2)
            
            self.dy, self.y = self.dy + self.ddy * self.dt, self.y + self.dy * self.dt
            
            current_state = {'pos': self.y.copy(), 'vel': self.dy.copy(), 'acc': self.ddy.copy()}
            state_history.append(current_state)
            
            for key, data in zip(['position', 'velocity', 'acceleration'], 
                                [current_state['pos'], current_state['vel'], current_state['acc']]):
                trajectory_data[key][step_idx] = data
            
            self.cs.step(tau, error)
        
        return trajectory_data['position']

# ==============================
# DMP Unit test
# ==============================
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Test canonical system
    cs = CanonicalSystem(dt=0.05)
    x_track = cs.rollout()
    plt.figure()
    plt.plot(x_track, label='Canonical x')
    plt.title('Canonical System Rollout')
    plt.xlabel('Timestep')
    plt.ylabel('x')
    plt.legend()

    # Test DMP behavior with a sine-wave trajectory
    dt = 0.01
    T = 1.0
    t = np.arange(0, T, dt)
    y_des = np.sin(2 * np.pi * 2 * t)

    dmp = DMP(n_dmps=1, n_bfs=50, dt=dt)
    y_interp = dmp.imitate(y_des)
    y_run = dmp.rollout()

    plt.figure()
    plt.plot(t, y_des, 'k--', label='Original')
    plt.plot(np.linspace(0, T, y_interp.shape[1]), y_interp.flatten(), 'b-.', label='Interpolated')
    plt.plot(np.linspace(0, T, y_run.shape[0]), y_run.flatten(), 'r-', label='DMP Rollout')
    plt.title('DMP Imitation and Rollout')
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()