#!/usr/bin/env python3
"""
Unified Behavioral Cloning Training System for Robotic Manipulation
Combines data loading and training into a single, streamlined implementation.
"""

import os
import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


@dataclass
class TrainingConfig:
    """Configuration parameters for the training process."""
    demo_directory: str = "demos"
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    validation_split: float = 0.2
    phase_loss_weight: float = 0.1
    observation_dims: int = 20
    action_dims: int = 7
    checkpoint_file: str = "bc_model.pth"
    num_workers: int = 4
    gradient_clip_norm: float = 1.0
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5


class RoboticManipulationDataset(Dataset):
    """Dataset for robotic manipulation demonstrations with phase labeling."""
    
    # Static peg position for NutAssemblySquare task
    PEG_POSITION = np.array([0.0, 0.0, 0.82])
    
    # Phase classification thresholds
    GRASP_DISTANCE_THRESHOLD = 0.05
    LIFT_HEIGHT_THRESHOLD = 0.05
    
    def __init__(self, demo_directory: str):
        """Initialize dataset by loading demonstration files."""
        self.demo_path = Path(demo_directory)
        self.demonstration_samples = []
        self._load_demonstrations()
        
    def _load_demonstrations(self) -> None:
        """Load all demonstration files and extract samples."""
        demo_files = self._find_demo_files()
        
        for demo_file in demo_files:
            try:
                demo_id = self._extract_demo_id(demo_file)
                demo_data = self._load_demo_file(demo_file)
                samples = self._extract_samples(demo_data, demo_id, demo_file.name)
                self.demonstration_samples.extend(samples)
            except Exception as e:
                print(f"Failed to process {demo_file.name}: {e}")
                continue
        
        if not self.demonstration_samples:
            raise ValueError(f"No valid samples found in {self.demo_path}")
    
    def _find_demo_files(self) -> List[Path]:
        """Find all demonstration pickle files."""
        demo_files = list(self.demo_path.glob("demo_*.pkl"))
        if not demo_files:
            raise FileNotFoundError(f"No demo files found in {self.demo_path}")
        return sorted(demo_files)
    
    def _extract_demo_id(self, demo_file: Path) -> str:
        """Extract demonstration ID from filename."""
        return demo_file.stem.split('_')[-1]
    
    def _load_demo_file(self, demo_file: Path) -> Dict[str, Any]:
        """Load demonstration data from pickle file."""
        with open(demo_file, 'rb') as f:
            return pickle.load(f)
    
    def _extract_samples(self, demo_data: Dict, demo_id: str, filename: str) -> List[Tuple]:
        """Extract individual samples from demonstration data."""
        try:
            # Primary key pattern
            key_patterns = {
                'eef_pos': f'demo_{demo_id}_obs_robot0_eef_pos',
                'eef_quat': f'demo_{demo_id}_obs_robot0_eef_quat',
                'object_state': f'demo_{demo_id}_obs_object',
                'actions': f'demo_{demo_id}_actions'
            }
            
            # Try to extract data with primary pattern
            extracted_data = self._try_extract_with_patterns(demo_data, key_patterns, demo_id)
            
            if not extracted_data:
                raise KeyError("Could not extract data with any key pattern")
            
            return self._create_samples_from_data(*extracted_data)
            
        except (KeyError, IndexError) as e:
            print(f"Warning: Skipping {filename} due to data extraction error: {e}")
            return []
    
    def _try_extract_with_patterns(self, demo_data: Dict, primary_keys: Dict, demo_id: str) -> Optional[Tuple]:
        """Try multiple key patterns to extract demonstration data."""
        # Try primary pattern
        if all(key in demo_data for key in primary_keys.values()):
            return self._extract_with_keys(demo_data, primary_keys)
        
        # Try nested pattern
        nested_source = demo_data.get(f'demo_{demo_id}', {})
        fallback_keys = {
            'eef_pos': 'obs_robot0_eef_pos',
            'eef_quat': 'obs_robot0_eef_quat',
            'object_state': 'obs_object',
            'actions': 'actions'
        }
        
        if all(key in nested_source for key in fallback_keys.values()):
            return self._extract_with_keys(nested_source, fallback_keys)
        
        # Try top-level fallback
        if all(key in demo_data for key in fallback_keys.values()):
            return self._extract_with_keys(demo_data, fallback_keys)
        
        return None
    
    def _extract_with_keys(self, data_source: Dict, key_mapping: Dict) -> Tuple:
        """Extract data arrays using specified key mapping."""
        eef_pos = data_source[key_mapping['eef_pos']]
        eef_quat = data_source[key_mapping['eef_quat']]
        object_state = data_source[key_mapping['object_state']]
        actions = data_source[key_mapping['actions']]
        
        # Parse object state (assumed format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z])
        obj_position = object_state[:, :3]
        obj_quaternion = object_state[:, 3:7]
        
        return eef_pos, eef_quat, obj_position, obj_quaternion, actions
    
    def _create_samples_from_data(self, eef_pos, eef_quat, obj_pos, obj_quat, actions) -> List[Tuple]:
        """Create individual samples from trajectory data."""
        samples = []
        peg_positions = np.tile(self.PEG_POSITION, (len(eef_pos), 1))
        
        for i in range(len(eef_pos)):
            sample = (
                eef_pos[i], eef_quat[i], obj_pos[i], 
                obj_quat[i], peg_positions[i], actions[i]
            )
            samples.append(sample)
        
        return samples
    
    def _compute_phase_label(self, obj_relative_pos: np.ndarray, action: np.ndarray, obj_height: float) -> int:
        """Determine manipulation phase based on state and action."""
        distance_to_object = np.linalg.norm(obj_relative_pos)
        gripper_active = action[-1] > 0
        
        if distance_to_object > self.GRASP_DISTANCE_THRESHOLD:
            return 0  # APPROACH phase
        elif distance_to_object <= self.GRASP_DISTANCE_THRESHOLD and not gripper_active:
            return 1  # GRASP phase
        elif gripper_active and obj_height < self.LIFT_HEIGHT_THRESHOLD:
            return 2  # LIFT phase
        else:
            return 3  # PLACE phase
    
    def _construct_observation(self, eef_pos, eef_quat, obj_pos, obj_quat, peg_pos) -> np.ndarray:
        """Construct the 20-dimensional observation vector."""
        obj_relative_pos = obj_pos - eef_pos
        peg_relative_pos = peg_pos - eef_pos
        
        distance_to_object = np.linalg.norm(obj_relative_pos)
        distance_to_peg = np.linalg.norm(peg_relative_pos)
        object_height = obj_pos[2] - self.PEG_POSITION[2]
        
        observation = np.concatenate([
            eef_pos,                    # End-effector position (3D)
            eef_quat,                   # End-effector quaternion (4D)
            obj_relative_pos,           # Object position relative to EEF (3D)
            obj_quat,                   # Object quaternion (4D)
            peg_relative_pos,           # Peg position relative to EEF (3D)
            [distance_to_object, distance_to_peg, object_height]  # Computed features (3D)
        ])
        
        return observation
    
    def __len__(self) -> int:
        return len(self.demonstration_samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset."""
        eef_pos, eef_quat, obj_pos, obj_quat, peg_pos, action = self.demonstration_samples[index]
        
        observation = self._construct_observation(eef_pos, eef_quat, obj_pos, obj_quat, peg_pos)
        obj_relative_pos = obj_pos - eef_pos
        object_height = obj_pos[2] - self.PEG_POSITION[2]
        phase = self._compute_phase_label(obj_relative_pos, action, object_height)
        
        return (
            torch.tensor(observation, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(phase, dtype=torch.long)
        )


class BehavioralCloningNetwork(nn.Module):
    """Neural network for behavioral cloning with multi-head output."""
    
    def __init__(self, observation_dim: int = 20, action_dim: int = 7):
        super().__init__()
        
        # Feature extraction backbone
        self.feature_extractor = nn.Sequential(
            self._create_layer(observation_dim, 512, dropout=0.1),
            self._create_layer(512, 256, dropout=0.1),
            self._create_layer(256, 128, dropout=0.0),
            self._create_layer(128, 64, dropout=0.0)
        )
        
        # Multi-head output layers
        self.action_heads = nn.ModuleDict({
            'position': nn.Linear(64, 3),
            'rotation': nn.Linear(64, 3),
            'gripper': nn.Linear(64, 1),
            'phase': nn.Linear(64, 4)
        })
    
    def _create_layer(self, in_features: int, out_features: int, dropout: float = 0.0) -> nn.Sequential:
        """Create a linear layer with ReLU activation and optional dropout."""
        layers = [nn.Linear(in_features, out_features), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
    
    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        features = self.feature_extractor(observations)
        
        # Compute action components
        position = self.action_heads['position'](features)
        rotation = self.action_heads['rotation'](features)
        gripper = torch.tanh(self.action_heads['gripper'](features))
        
        # Combine action components
        actions = torch.cat([position, rotation, gripper], dim=-1)
        
        # Compute phase logits
        phase_logits = self.action_heads['phase'](features)
        
        return actions, phase_logits


class DataNormalizer:
    """Handles normalization of observations and actions."""
    
    def __init__(self, observations: torch.Tensor, actions: torch.Tensor):
        self.obs_mean = torch.mean(observations, dim=0)
        self.obs_std = torch.std(observations, dim=0)
        self.act_mean = torch.mean(actions, dim=0)
        self.act_std = torch.std(actions, dim=0)
        
        # Prevent division by zero
        self.obs_std = torch.clamp(self.obs_std, min=1e-7)
        self.act_std = torch.clamp(self.act_std, min=1e-7)
    
    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Normalize observations using computed statistics."""
        return (observations - self.obs_mean) / self.obs_std
    
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions using computed statistics."""
        return (actions - self.act_mean) / self.act_std
    
    def to_device(self, device: torch.device) -> 'DataNormalizer':
        """Move normalization statistics to specified device."""
        self.obs_mean = self.obs_mean.to(device)
        self.obs_std = self.obs_std.to(device)
        self.act_mean = self.act_mean.to(device)
        self.act_std = self.act_std.to(device)
        return self
    
    def get_cpu_stats(self) -> Dict[str, torch.Tensor]:
        """Get normalization statistics as CPU tensors."""
        return {
            'obs_mean': self.obs_mean.cpu(),
            'obs_std': self.obs_std.cpu(),
            'act_mean': self.act_mean.cpu(),
            'act_std': self.act_std.cpu()
        }


class BehavioralCloningTrainer:
    """Main trainer class for behavioral cloning."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_validation_loss = float('inf')
        
        # Initialize components
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.normalizer = None
        
    def setup_data(self) -> None:
        """Set up dataset and data loaders."""
        print("Loading demonstration dataset...")
        self.dataset = RoboticManipulationDataset(self.config.demo_directory)
        
        if len(self.dataset) == 0:
            raise ValueError("Dataset contains no samples")
        
        # Split dataset
        val_size = int(len(self.dataset) * self.config.validation_split)
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        
        # Compute normalization statistics
        self._compute_normalization_stats()
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_batch
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_batch
        )
    
    def _compute_normalization_stats(self) -> None:
        """Compute normalization statistics from training data."""
        print("Computing normalization statistics...")
        
        temp_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_batch
        )
        
        all_observations = []
        all_actions = []
        
        for obs_batch, act_batch, _ in temp_loader:
            all_observations.append(obs_batch)
            all_actions.append(act_batch)
        
        if not all_observations:
            raise ValueError("No data available for computing normalization statistics")
        
        observations = torch.cat(all_observations, dim=0)
        actions = torch.cat(all_actions, dim=0)
        
        self.normalizer = DataNormalizer(observations, actions)
        print(f"Normalization stats computed from {len(observations)} samples")
    
    def _collate_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate function for batch processing."""
        observations, actions, phases = zip(*batch)
        return (
            torch.stack(observations),
            torch.stack(actions),
            torch.tensor(phases, dtype=torch.long)
        )
    
    def setup_model(self) -> None:
        """Initialize model, optimizer, and scheduler."""
        print(f"Setting up model on device: {self.device}")
        
        self.model = BehavioralCloningNetwork(
            self.config.observation_dims,
            self.config.action_dims
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.config.scheduler_patience,
            factor=self.config.scheduler_factor
        )
        
        # Move normalizer to device
        self.normalizer.to_device(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = defaultdict(float)
        batch_count = 0
        
        for obs_batch, act_batch, phase_batch in self.train_loader:
            # Move to device and normalize
            obs_batch = self.normalizer.normalize_observations(obs_batch.to(self.device))
            act_batch_normalized = self.normalizer.normalize_actions(act_batch.to(self.device))
            phase_batch = phase_batch.to(self.device)
            
            # Forward pass
            pred_actions, pred_phases = self.model(obs_batch)
            
            # Compute losses
            action_loss = F.mse_loss(pred_actions, act_batch_normalized)
            phase_loss = F.cross_entropy(pred_phases, phase_batch)
            total_loss = action_loss + self.config.phase_loss_weight * phase_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            
            # Update metrics
            metrics['action_loss'] += action_loss.item()
            metrics['phase_loss'] += phase_loss.item()
            metrics['total_loss'] += total_loss.item()
            batch_count += 1
        
        # Average metrics
        return {key: value / batch_count for key, value in metrics.items()}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        metrics = defaultdict(float)
        batch_count = 0
        
        with torch.no_grad():
            for obs_batch, act_batch, phase_batch in self.val_loader:
                # Move to device and normalize
                obs_batch = self.normalizer.normalize_observations(obs_batch.to(self.device))
                act_batch_normalized = self.normalizer.normalize_actions(act_batch.to(self.device))
                phase_batch = phase_batch.to(self.device)
                
                # Forward pass
                pred_actions, pred_phases = self.model(obs_batch)
                
                # Compute losses
                action_loss = F.mse_loss(pred_actions, act_batch_normalized)
                phase_loss = F.cross_entropy(pred_phases, phase_batch)
                total_loss = action_loss + self.config.phase_loss_weight * phase_loss
                
                # Update metrics
                metrics['action_loss'] += action_loss.item()
                metrics['phase_loss'] += phase_loss.item()
                metrics['total_loss'] += total_loss.item()
                batch_count += 1
        
        # Average metrics
        return {key: value / batch_count for key, value in metrics.items()}
    
    def save_checkpoint(self, epoch: int, validation_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'validation_loss': validation_loss,
            'config': self.config,
            **self.normalizer.get_cpu_stats()
        }
        
        torch.save(checkpoint_data, self.config.checkpoint_file)
        print(f"Checkpoint saved: {self.config.checkpoint_file}")
    
    def train(self) -> None:
        """Main training loop."""
        print("Starting behavioral cloning training...")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_metrics['total_loss'])
            
            # Save best model
            if val_metrics['total_loss'] < self.best_validation_loss:
                self.best_validation_loss = val_metrics['total_loss']
                self.save_checkpoint(epoch, val_metrics['total_loss'])
                print(f"Epoch {epoch + 1}: New best validation loss: {self.best_validation_loss:.6f}")
            
            # Periodic logging
            if (epoch + 1) % 10 == 0 or epoch == self.config.epochs - 1:
                self._log_epoch_results(epoch + 1, train_metrics, val_metrics)
        
        print(f"Training completed. Best model saved to {self.config.checkpoint_file}")
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log training results for an epoch."""
        print(f"\nEpoch {epoch}/{self.config.epochs}")
        print(f"  Train - Total: {train_metrics['total_loss']:.6f}, "
              f"Action: {train_metrics['action_loss']:.6f}, "
              f"Phase: {train_metrics['phase_loss']:.6f}")
        print(f"  Val   - Total: {val_metrics['total_loss']:.6f}, "
              f"Action: {val_metrics['action_loss']:.6f}, "
              f"Phase: {val_metrics['phase_loss']:.6f}")
        print(f"  Best Val Loss: {self.best_validation_loss:.6f}")
        print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")


def main():
    """Main execution function."""
    config = TrainingConfig()
    trainer = BehavioralCloningTrainer(config)
    
    try:
        trainer.setup_data()
        trainer.setup_model()
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()