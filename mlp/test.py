#!/usr/bin/env python3
"""
BC Policy Wrapper for the unified behavioral cloning training system.
Provides inference interface for trained models in robosuite environments.
"""

import torch
import numpy as np
import robosuite as suite
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Import from the unified training system
from train import BehavioralCloningNetwork, TrainingConfig


class BCPolicyInference:
    """Inference wrapper for trained behavioral cloning models."""
    
    # Static peg position for NutAssemblySquare task
    PEG_POSITION = np.array([0.0, 0.0, 0.82])
    
    def __init__(self, checkpoint_path: str = 'bc_model.pth'):
        """Initialize the BC policy inference system."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load and initialize model
        self._load_checkpoint()
        self._initialize_model()
        
        print(f"BC Policy loaded successfully from {checkpoint_path}")
        print(f"Running on device: {self.device}")
    
    def _load_checkpoint(self) -> None:
        """Load model checkpoint and extract components."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        print(f"Loading checkpoint from {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract normalization statistics
        self.normalization_stats = {
            'obs_mean': self.checkpoint['obs_mean'].to(self.device),
            'obs_std': self.checkpoint['obs_std'].to(self.device),
            'act_mean': self.checkpoint['act_mean'].to(self.device),
            'act_std': self.checkpoint['act_std'].to(self.device)
        }
        
        # Extract model configuration
        self.model_config = self.checkpoint.get('config', None)
        if self.model_config is None:
            # Fallback to default configuration
            print("Warning: No config found in checkpoint, using defaults")
            self.model_config = TrainingConfig()
    
    def _initialize_model(self) -> None:
        """Initialize and load the trained model."""
        # Create model with same architecture as training
        self.model = BehavioralCloningNetwork(
            observation_dim=self.model_config.observation_dims,
            action_dim=self.model_config.action_dims
        ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model initialized with {self.model_config.observation_dims}D obs, "
              f"{self.model_config.action_dims}D action")
    
    def construct_observation_vector(self, robosuite_obs: Dict[str, Any]) -> np.ndarray:
        """
        Construct 20D observation vector from robosuite observation dictionary.
        
        Args:
            robosuite_obs: Dictionary containing robosuite observation data
            
        Returns:
            20-dimensional observation vector as numpy array
        """
        # Extract components from robosuite observation
        end_effector_pos = robosuite_obs['robot0_eef_pos']
        end_effector_quat = robosuite_obs['robot0_eef_quat']
        object_position = robosuite_obs['SquareNut_pos']
        object_quaternion = robosuite_obs['SquareNut_quat']
        
        # Compute relative positions
        obj_relative_pos = object_position - end_effector_pos
        peg_relative_pos = self.PEG_POSITION - end_effector_pos
        
        # Compute derived features
        distance_to_object = np.linalg.norm(obj_relative_pos)
        distance_to_peg = np.linalg.norm(peg_relative_pos)
        object_height = object_position[2] - self.PEG_POSITION[2]
        
        # Construct full observation vector (20D)
        observation_vector = np.concatenate([
            end_effector_pos,           # End-effector position (3D)
            end_effector_quat,          # End-effector quaternion (4D)
            obj_relative_pos,           # Object position relative to EEF (3D)
            object_quaternion,          # Object quaternion (4D)
            peg_relative_pos,           # Peg position relative to EEF (3D)
            [distance_to_object, distance_to_peg, object_height]  # Computed features (3D)
        ])
        
        return observation_vector
    
    def predict_action(self, robosuite_obs: Dict[str, Any]) -> np.ndarray:
        """
        Predict action given robosuite observation.
        
        Args:
            robosuite_obs: Dictionary containing robosuite observation data
            
        Returns:
            7-dimensional action vector as numpy array
        """
        # Construct observation vector
        obs_vector = self.construct_observation_vector(robosuite_obs)
        
        # Convert to tensor and add batch dimension
        obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Normalize observation
        normalized_obs = self._normalize_observation(obs_tensor)
        
        # Predict normalized action
        with torch.no_grad():
            predicted_action_norm, _ = self.model(normalized_obs)
        
        # Denormalize action
        denormalized_action = self._denormalize_action(predicted_action_norm)
        
        # Convert to numpy and remove batch dimension
        action = denormalized_action.cpu().numpy().squeeze(0)
        
        return action
    
    def _normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize observation using training statistics."""
        return (observation - self.normalization_stats['obs_mean']) / self.normalization_stats['obs_std']
    
    def _denormalize_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        """Denormalize action using training statistics."""
        return normalized_action * self.normalization_stats['act_std'] + self.normalization_stats['act_mean']
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'checkpoint_path': str(self.checkpoint_path),
            'device': str(self.device),
            'epoch': self.checkpoint.get('epoch', 'unknown'),
            'validation_loss': self.checkpoint.get('validation_loss', 'unknown'),
            'observation_dims': self.model_config.observation_dims,
            'action_dims': self.model_config.action_dims,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }


class RobosuiteEvaluator:
    """Evaluator for BC policies in robosuite environments."""
    
    def __init__(self, policy: BCPolicyInference, env_config: Optional[Dict] = None):
        """
        Initialize evaluator with policy and environment configuration.
        
        Args:
            policy: Trained BC policy for inference
            env_config: Optional environment configuration parameters
        """
        self.policy = policy
        self.env_config = env_config or self._get_default_env_config()
        self.environment = None
        
    def _get_default_env_config(self) -> Dict[str, Any]:
        """Get default environment configuration."""
        return {
            'env_name': 'NutAssemblySquare',
            'robots': 'Panda',
            'has_renderer': True,
            'has_offscreen_renderer': False,
            'use_camera_obs': False,
            'ignore_done': True,
            'horizon': 500
        }
    
    def setup_environment(self) -> None:
        """Initialize the robosuite environment."""
        print("Setting up robosuite environment...")
        self.environment = suite.make(**self.env_config)
        print(f"Environment '{self.env_config['env_name']}' ready")
    
    def run_episode(self, episode_id: int, max_steps: int = 500, verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single evaluation episode.
        
        Args:
            episode_id: Episode identifier for logging
            max_steps: Maximum number of steps per episode
            verbose: Whether to print episode progress
            
        Returns:
            Dictionary containing episode results
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized. Call setup_environment() first.")
        
        # Reset environment
        observation = self.environment.reset()
        
        if verbose:
            object_pos = observation['SquareNut_pos']
            print(f"Episode {episode_id}: SquareNut initial position: {object_pos}")
        
        # Episode tracking
        episode_reward = 0
        episode_success = False
        step_count = 0
        
        # Run episode
        for step in range(max_steps):
            # Get action from policy
            action = self.policy.predict_action(observation)
            
            # Execute action
            observation, reward, done, info = self.environment.step(action)
            episode_reward += reward
            step_count += 1
            
            # Render if enabled
            if self.env_config.get('has_renderer', False):
                self.environment.render()
            
            # Check for success
            if reward == 1.0:
                episode_success = True
                if verbose:
                    print(f"Episode {episode_id}: SUCCESS after {step_count} steps!")
                break
        
        # Episode results
        results = {
            'episode_id': episode_id,
            'success': episode_success,
            'total_reward': episode_reward,
            'steps': step_count,
            'final_object_pos': observation['SquareNut_pos'].copy()
        }
        
        if verbose and not episode_success:
            print(f"Episode {episode_id}: Failed after {step_count} steps")
        
        return results
    
    def evaluate_policy(self, num_episodes: int = 10, max_steps_per_episode: int = 500) -> Dict[str, Any]:
        """
        Evaluate policy over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"Starting evaluation over {num_episodes} episodes...")
        
        if self.environment is None:
            self.setup_environment()
        
        # Run episodes
        episode_results = []
        for episode_idx in range(num_episodes):
            result = self.run_episode(
                episode_id=episode_idx + 1,
                max_steps=max_steps_per_episode,
                verbose=True
            )
            episode_results.append(result)
        
        # Compute aggregate statistics
        successes = sum(1 for result in episode_results if result['success'])
        success_rate = successes / num_episodes
        
        successful_episodes = [r for r in episode_results if r['success']]
        avg_success_steps = np.mean([r['steps'] for r in successful_episodes]) if successful_episodes else 0
        
        total_rewards = [r['total_reward'] for r in episode_results]
        avg_reward = np.mean(total_rewards)
        
        # Summary results
        evaluation_summary = {
            'num_episodes': num_episodes,
            'success_count': successes,
            'success_rate': success_rate,
            'average_reward': avg_reward,
            'average_success_steps': avg_success_steps,
            'episode_results': episode_results
        }
        
        # Print summary
        print(f"\n=== Evaluation Summary ===")
        print(f"Episodes: {num_episodes}")
        print(f"Successes: {successes}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Reward: {avg_reward:.3f}")
        if successful_episodes:
            print(f"Average Steps to Success: {avg_success_steps:.1f}")
        
        return evaluation_summary


def main():
    """Main evaluation function."""
    try:
        # Initialize BC policy
        print("Initializing BC Policy...")
        bc_policy = BCPolicyInference('bc_model.pth')
        
        # Print model information
        model_info = bc_policy.get_model_info()
        print(f"\n=== Model Information ===")
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
        # Initialize evaluator
        evaluator = RobosuiteEvaluator(bc_policy)
        
        # Run evaluation
        results = evaluator.evaluate_policy(num_episodes=10, max_steps_per_episode=500)
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()