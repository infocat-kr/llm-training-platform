"""
훈련 콜백들
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Callable
import json
from abc import ABC, abstractmethod


class Callback(ABC):
    """콜백 기본 클래스"""
    
    @abstractmethod
    def __call__(self, trainer, metrics: Dict[str, float]):
        """콜백 실행"""
        pass


class EarlyStopping(Callback):
    """조기 종료 콜백"""
    
    def __init__(self, 
                 monitor: str = 'val_loss',
                 patience: int = 5,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best_weights: bool = True):
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            
    def __call__(self, trainer, metrics: Dict[str, float]):
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            return
            
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(trainer)
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.counter = 0
            self.save_checkpoint(trainer)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
                if self.restore_best_weights and self.best_weights is not None:
                    trainer.model.load_state_dict(self.best_weights)
                trainer.should_stop = True
                
    def save_checkpoint(self, trainer):
        """최고 성능 모델 저장"""
        if self.restore_best_weights:
            self.best_weights = trainer.model.state_dict().copy()


class ModelCheckpoint(Callback):
    """모델 체크포인트 콜백"""
    
    def __init__(self, 
                 filepath: str,
                 monitor: str = 'val_loss',
                 save_best_only: bool = True,
                 mode: str = 'min',
                 save_freq: int = 1,
                 save_optimizer: bool = True,
                 save_scheduler: bool = True):
        
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.best_score = None
        self.epoch_count = 0
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
            
    def __call__(self, trainer, metrics: Dict[str, float]):
        self.epoch_count += 1
        
        # 주기적 저장
        if not self.save_best_only and self.epoch_count % self.save_freq == 0:
            self.save_model(trainer, f"epoch_{self.epoch_count}")
            return
            
        # 최고 성능 모델 저장
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            return
            
        if self.best_score is None or self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.save_model(trainer, "best_model")
            
    def save_model(self, trainer, suffix: str = ""):
        """모델 저장"""
        if suffix:
            filename = f"{self.filepath}_{suffix}.pt"
        else:
            filename = f"{self.filepath}.pt"
            
        checkpoint = {
            'model_state_dict': trainer.model.state_dict(),
            'epoch': trainer.epoch,
            'global_step': trainer.global_step,
            'best_val_loss': trainer.best_val_loss
        }
        
        if self.save_optimizer:
            checkpoint['optimizer_state_dict'] = trainer.optimizer.state_dict()
            
        if self.save_scheduler and trainer.scheduler is not None:
            checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()
            
        torch.save(checkpoint, filename)
        print(f"Model saved to {filename}")


class LearningRateScheduler(Callback):
    """학습률 스케줄러 콜백"""
    
    def __init__(self, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 factor: float = 0.1,
                 patience: int = 3,
                 min_lr: float = 1e-7):
        
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        
        self.best_score = None
        self.counter = 0
        
        if mode == 'min':
            self.monitor_op = np.less
        else:
            self.monitor_op = np.greater
            
    def __call__(self, trainer, metrics: Dict[str, float]):
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            return
            
        if self.best_score is None:
            self.best_score = current_score
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.scheduler.step()
                self.counter = 0
                print(f"Learning rate reduced to {trainer.optimizer.param_groups[0]['lr']}")


class MetricsLogger(Callback):
    """메트릭 로거 콜백"""
    
    def __init__(self, log_file: str = "training_metrics.json"):
        self.log_file = log_file
        self.metrics_history = []
        
    def __call__(self, trainer, metrics: Dict[str, float]):
        # 메트릭에 에포크 정보 추가
        log_entry = {
            'epoch': trainer.epoch,
            'global_step': trainer.global_step,
            **metrics
        }
        
        self.metrics_history.append(log_entry)
        
        # 파일에 저장
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class GradientMonitor(Callback):
    """그래디언트 모니터링 콜백"""
    
    def __init__(self, log_freq: int = 100):
        self.log_freq = log_freq
        
    def __call__(self, trainer, metrics: Dict[str, float]):
        if trainer.global_step % self.log_freq == 0:
            total_norm = 0
            param_count = 0
            
            for param in trainer.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
            total_norm = total_norm ** (1. / 2)
            
            print(f"Global gradient norm: {total_norm:.4f}")
            
            if hasattr(trainer, 'use_wandb') and trainer.config.use_wandb:
                import wandb
                wandb.log({
                    'gradient_norm': total_norm,
                    'global_step': trainer.global_step
                })


class ModelSummary(Callback):
    """모델 요약 콜백"""
    
    def __init__(self, log_freq: int = 1000):
        self.log_freq = log_freq
        
    def __call__(self, trainer, metrics: Dict[str, float]):
        if trainer.global_step % self.log_freq == 0:
            total_params = sum(p.numel() for p in trainer.model.parameters())
            trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            
            print(f"Model Summary:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "  Memory usage: N/A (CPU)")


class CustomCallback(Callback):
    """커스텀 콜백"""
    
    def __init__(self, callback_func: Callable):
        self.callback_func = callback_func
        
    def __call__(self, trainer, metrics: Dict[str, float]):
        self.callback_func(trainer, metrics)


class CallbackManager:
    """콜백 관리자"""
    
    def __init__(self, callbacks: Optional[list] = None):
        self.callbacks = callbacks or []
        
    def add_callback(self, callback: Callback):
        """콜백 추가"""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback: Callback):
        """콜백 제거"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            
    def __call__(self, trainer, metrics: Dict[str, float]):
        """모든 콜백 실행"""
        for callback in self.callbacks:
            callback(trainer, metrics)
