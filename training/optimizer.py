"""
옵티마이저 및 스케줄러 유틸리티
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau,
    CosineAnnealingWarmRestarts, OneCycleLR, LinearLR, ConstantLR
)
from typing import Dict, Any, Optional, Union
import math


def create_optimizer(model: torch.nn.Module, 
                    optimizer_type: str = 'adamw',
                    learning_rate: float = 1e-4,
                    weight_decay: float = 0.01,
                    betas: tuple = (0.9, 0.999),
                    eps: float = 1e-8,
                    **kwargs) -> torch.optim.Optimizer:
    """옵티마이저 생성"""
    
    if optimizer_type.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **kwargs
        )
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adafactor':
        # Adafactor는 별도 구현 필요
        return create_adafactor_optimizer(
            model.parameters(),
            lr=learning_rate,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def create_adafactor_optimizer(parameters, lr=1e-3, eps2=1e-30, clip_threshold=1.0, 
                              decay_rate=-0.8, beta1=None, weight_decay=0.0, 
                              scale_parameter=True, relative_step_size=True, 
                              warmup_init=False):
    """Adafactor 옵티마이저 (간단한 구현)"""
    # 실제로는 transformers 라이브러리의 Adafactor를 사용하는 것이 좋습니다
    return optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = 'cosine',
                    num_training_steps: Optional[int] = None,
                    num_warmup_steps: Optional[int] = None,
                    **kwargs) -> torch.optim.lr_scheduler._LRScheduler:
    """학습률 스케줄러 생성"""
    
    if scheduler_type.lower() == 'constant':
        return ConstantLR(optimizer, factor=1.0, total_iters=0)
        
    elif scheduler_type.lower() == 'linear':
        if num_training_steps is None:
            raise ValueError("num_training_steps required for linear scheduler")
        return LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 0.1),
            end_factor=kwargs.get('end_factor', 1.0),
            total_iters=num_training_steps,
            **kwargs
        )
        
    elif scheduler_type.lower() == 'cosine':
        if num_training_steps is None:
            raise ValueError("num_training_steps required for cosine scheduler")
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=kwargs.get('eta_min', 0),
            **kwargs
        )
        
    elif scheduler_type.lower() == 'cosine_warm_restarts':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 1),
            eta_min=kwargs.get('eta_min', 0),
            **kwargs
        )
        
    elif scheduler_type.lower() == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
            **kwargs
        )
        
    elif scheduler_type.lower() == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95),
            **kwargs
        )
        
    elif scheduler_type.lower() == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            **kwargs
        )
        
    elif scheduler_type.lower() == 'one_cycle':
        if num_training_steps is None:
            raise ValueError("num_training_steps required for one_cycle scheduler")
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr']),
            total_steps=num_training_steps,
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos'),
            **kwargs
        )
        
    elif scheduler_type.lower() == 'warmup_cosine':
        if num_training_steps is None or num_warmup_steps is None:
            raise ValueError("num_training_steps and num_warmup_steps required for warmup_cosine scheduler")
        return WarmupCosineScheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
        
    elif scheduler_type.lower() == 'warmup_linear':
        if num_training_steps is None or num_warmup_steps is None:
            raise ValueError("num_training_steps and num_warmup_steps required for warmup_linear scheduler")
        return WarmupLinearScheduler(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **kwargs
        )
        
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 스케줄러"""
    
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, 
                 min_lr_ratio: float = 0.0):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        # 현재 스텝 계산 (외부에서 관리)
        pass
        
    def get_lr(self, step: int):
        if step < self.num_warmup_steps:
            # Warmup phase
            return [base_lr * step / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            progress = min(progress, 1.0)
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * 
                             0.5 * (1 + math.cos(math.pi * progress))) 
                   for base_lr in self.base_lrs]
                   
    def update_lr(self, step: int):
        lrs = self.get_lr(step)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


class WarmupLinearScheduler:
    """Warmup + Linear Decay 스케줄러"""
    
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, 
                 min_lr_ratio: float = 0.0):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        # 현재 스텝 계산 (외부에서 관리)
        pass
        
    def get_lr(self, step: int):
        if step < self.num_warmup_steps:
            # Warmup phase
            return [base_lr * step / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            # Linear decay phase
            progress = (step - self.num_warmup_steps) / (self.num_training_steps - self.num_warmup_steps)
            progress = min(progress, 1.0)
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - progress))
                   for base_lr in self.base_lrs]
                   
    def update_lr(self, step: int):
        lrs = self.get_lr(step)
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr


class OptimizerConfig:
    """옵티마이저 설정 클래스"""
    
    def __init__(self,
                 optimizer_type: str = 'adamw',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 betas: tuple = (0.9, 0.999),
                 eps: float = 1e-8,
                 scheduler_type: str = 'cosine',
                 num_training_steps: Optional[int] = None,
                 num_warmup_steps: Optional[int] = None,
                 **kwargs):
        
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.scheduler_type = scheduler_type
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.kwargs = kwargs
        
    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """옵티마이저 생성"""
        return create_optimizer(
            model,
            optimizer_type=self.optimizer_type,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
            **self.kwargs
        )
        
    def create_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """스케줄러 생성"""
        return create_scheduler(
            optimizer,
            scheduler_type=self.scheduler_type,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.num_warmup_steps,
            **self.kwargs
        )
