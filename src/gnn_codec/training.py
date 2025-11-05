"""Training utilities for GNN-based codec-holography engine."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from typing import List, Dict, Optional, Tuple
import time
import os
from .engine import GNNCodecHolographyEngine


class GNNTrainingSystem:
    """Training system for GNN-based codec-holography engine.
    
    Args:
        engine: The codec engine to train
        device: Device to run training on
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
    """
    
    def __init__(
        self,
        engine: GNNCodecHolographyEngine,
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        self.engine = engine.to(device)
        self.device = device
        
        # Optimizer setup
        self.optimizer = optim.AdamW(
            engine.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'compression_ratio': [],
            'learning_rate': [],
            'epoch_time': []
        }
    
    def setup_scheduler(self, scheduler_type: str = 'cosine', **kwargs):
        """Setup learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('cosine', 'step')
            **kwargs: Additional scheduler parameters
        """
        if scheduler_type == 'cosine':
            T_max = kwargs.get('T_max', 100)
            eta_min = kwargs.get('eta_min', 1e-6)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'step':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def train_on_weights(
        self,
        weight_list: List[torch.Tensor],
        epochs: int = 50,
        batch_size: int = 4,
        validation_split: float = 0.1,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """Train the codec engine on a list of weight tensors.
        
        Args:
            weight_list: List of weight tensors to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            checkpoint_dir: Directory to save checkpoints
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        # Split data
        val_size = int(len(weight_list) * validation_split)
        train_weights = weight_list[val_size:]
        val_weights = weight_list[:val_size] if val_size > 0 else []
        
        if verbose:
            print(f"Training on {len(train_weights)} weights, validating on {len(val_weights)}")
        
        # Setup scheduler if not already done
        if self.scheduler is None:
            self.setup_scheduler('cosine', T_max=epochs)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_compression = self._train_epoch(
                train_weights, batch_size, verbose and epoch % 10 == 0
            )
            
            # Validation phase
            if val_weights:
                val_loss, val_compression = self._validate_epoch(val_weights, batch_size)
            else:
                val_loss, val_compression = train_loss, train_compression
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['compression_ratio'].append(train_compression)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                if checkpoint_dir:
                    self._save_checkpoint(checkpoint_dir, epoch, val_loss, 'best')
            
            # Progress reporting
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1:3d}/{epochs}: "
                      f"Loss={train_loss:.6f}, "
                      f"Compression={train_compression:.1f}x, "
                      f"LR={current_lr:.2e}, "
                      f"Time={epoch_time:.2f}s")
            
            # Periodic checkpoints
            if checkpoint_dir and epoch % 20 == 0:
                self._save_checkpoint(checkpoint_dir, epoch, val_loss, 'periodic')
        
        return self.history
    
    def _train_epoch(
        self,
        weight_list: List[torch.Tensor],
        batch_size: int,
        verbose: bool = False
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.engine.train()
        
        total_loss = 0.0
        total_compression = 0.0
        num_batches = 0
        
        for i in range(0, len(weight_list), batch_size):
            batch_weights = weight_list[i:i + batch_size]
            
            batch_loss = 0.0
            batch_compression = 0.0
            
            for weights in batch_weights:
                weights = weights.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                encoded = self.engine(weights)
                decoded = self.engine.decode(encoded)
                
                # Compute loss
                loss = self.engine.compute_loss(weights, decoded, encoded)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.engine.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate statistics
                batch_loss += loss.item()
                if encoded.get('success', False):
                    compression_info = self.engine.get_compression_info(encoded)
                    batch_compression += compression_info['compression_ratio']
            
            total_loss += batch_loss / len(batch_weights)
            total_compression += batch_compression / len(batch_weights)
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_compression = total_compression / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_compression
    
    def _validate_epoch(
        self,
        weight_list: List[torch.Tensor],
        batch_size: int
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.engine.eval()
        
        total_loss = 0.0
        total_compression = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for i in range(0, len(weight_list), batch_size):
                batch_weights = weight_list[i:i + batch_size]
                
                for weights in batch_weights:
                    weights = weights.to(self.device)
                    
                    # Forward pass
                    encoded = self.engine(weights)
                    decoded = self.engine.decode(encoded)
                    
                    # Compute loss
                    loss = self.engine.compute_loss(weights, decoded, encoded)
                    
                    total_loss += loss.item()
                    if encoded.get('success', False):
                        compression_info = self.engine.get_compression_info(encoded)
                        total_compression += compression_info['compression_ratio']
                    
                    num_samples += 1
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        avg_compression = total_compression / num_samples if num_samples > 0 else 0.0
        
        return avg_loss, avg_compression
    
    def _save_checkpoint(self, checkpoint_dir: str, epoch: int, loss: float, suffix: str):
        """Save model checkpoint."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.engine.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filename = f"checkpoint_{suffix}.pth"
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.engine.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0)
        }
    
    def evaluate_compression_performance(
        self,
        test_weights: List[torch.Tensor],
        detailed: bool = True
    ) -> Dict:
        """Evaluate compression performance on test set.
        
        Args:
            test_weights: List of weight tensors for evaluation
            detailed: Whether to return detailed per-weight statistics
            
        Returns:
            Performance evaluation results
        """
        self.engine.eval()
        
        results = {
            'compression_ratios': [],
            'mse_errors': [],
            'memory_savings': [],
            'processing_times': []
        }
        
        detailed_results = [] if detailed else None
        
        with torch.no_grad():
            for i, weights in enumerate(test_weights):
                weights = weights.to(self.device)
                
                start_time = time.time()
                
                # Encode and decode
                encoded = self.engine(weights)
                decoded = self.engine.decode(encoded)
                
                processing_time = (time.time() - start_time) * 1000  # milliseconds
                
                # Compute metrics
                mse_error = torch.mean((weights - decoded) ** 2).item()
                
                if encoded.get('success', False):
                    compression_info = self.engine.get_compression_info(encoded)
                    compression_ratio = compression_info['compression_ratio']
                    memory_saving = compression_info['memory_savings_percent']
                else:
                    compression_ratio = 1.0
                    memory_saving = 0.0
                
                # Record results
                results['compression_ratios'].append(compression_ratio)
                results['mse_errors'].append(mse_error)
                results['memory_savings'].append(memory_saving)
                results['processing_times'].append(processing_time)
                
                if detailed:
                    detailed_results.append({
                        'index': i,
                        'shape': weights.shape,
                        'compression_ratio': compression_ratio,
                        'mse_error': mse_error,
                        'memory_saving_percent': memory_saving,
                        'processing_time_ms': processing_time
                    })
        
        # Compute summary statistics
        summary = {
            'mean_compression': float(torch.tensor(results['compression_ratios']).mean()),
            'std_compression': float(torch.tensor(results['compression_ratios']).std()),
            'mean_mse': float(torch.tensor(results['mse_errors']).mean()),
            'mean_memory_saving': float(torch.tensor(results['memory_savings']).mean()),
            'mean_processing_time': float(torch.tensor(results['processing_times']).mean()),
            'total_weights_tested': len(test_weights)
        }
        
        return {
            'summary': summary,
            'detailed': detailed_results if detailed else None,
            'raw_results': results
        }