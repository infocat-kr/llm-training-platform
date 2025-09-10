"""
모델 훈련 클래스
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import time
import os
from typing import Dict, Any, Optional, List, Callable
import wandb
from tqdm import tqdm
import json

from .loss import LanguageModelLoss, ClassificationLoss
from .metrics import calculate_metrics, perplexity
from .callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


class TrainingConfig:
    """훈련 설정 클래스"""
    
    def __init__(self,
                 epochs: int = 10,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 1000,
                 max_grad_norm: float = 1.0,
                 accumulation_steps: int = 1,
                 eval_steps: int = 500,
                 save_steps: int = 1000,
                 logging_steps: int = 100,
                 output_dir: str = './outputs',
                 use_wandb: bool = False,
                 wandb_project: str = 'llm-training',
                 device: str = 'auto',
                 mixed_precision: bool = False,
                 gradient_checkpointing: bool = False):
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)


class Trainer:
    """모델 훈련 클래스"""
    
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 loss_fn: Optional[nn.Module] = None,
                 optimizer: Optional[Optimizer] = None,
                 scheduler: Optional[_LRScheduler] = None,
                 callbacks: Optional[List[Callable]] = None):
        
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 디바이스 설정
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
            
        self.model.to(self.device)
        
        # 손실 함수
        if loss_fn is None:
            self.loss_fn = LanguageModelLoss()
        else:
            self.loss_fn = loss_fn
            
        # 옵티마이저
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer
            
        # 스케줄러
        self.scheduler = scheduler
        
        # 콜백
        self.callbacks = callbacks or []
        
        # Mixed precision
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # 훈련 상태
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Wandb 초기화
        if config.use_wandb:
            wandb.init(project=config.wandb_project)
            wandb.watch(self.model)
            
    def train(self):
        """훈련 실행"""
        print(f"Training started on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # 훈련
            train_metrics = self._train_epoch()
            
            # 검증
            if self.val_loader is not None:
                val_metrics = self._validate_epoch()
                
                # 콜백 실행
                for callback in self.callbacks:
                    callback(self, val_metrics)
                    
                # 모델 저장
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self._save_model('best_model.pt')
                    
            # 에포크별 모델 저장
            self._save_model(f'epoch_{epoch + 1}.pt')
            
            # 히스토리 저장
            epoch_metrics = {'epoch': epoch + 1, **train_metrics}
            if self.val_loader is not None:
                epoch_metrics.update(val_metrics)
            self.training_history.append(epoch_metrics)
            
            # Wandb 로깅
            if self.config.use_wandb:
                wandb.log(epoch_metrics)
                
        print("Training completed!")
        
        # 최종 모델 저장
        self._save_model('final_model.pt')
        self._save_training_history()
        
    def _train_epoch(self) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0
        total_tokens = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 배치를 디바이스로 이동
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = self.loss_fn(outputs, batch)
            else:
                outputs = self.model(**batch)
                loss = self.loss_fn(outputs, batch)
                
            # Gradient accumulation
            loss = loss / self.config.accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
            # Optimizer step
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                if self.scheduler is not None:
                    self.scheduler.step()
                    
                self.optimizer.zero_grad()
                self.global_step += 1
                
            # 메트릭 계산
            total_loss += loss.item() * self.config.accumulation_steps
            if 'labels' in batch:
                total_tokens += (batch['labels'] != -100).sum().item()
            num_batches += 1
            
            # 로깅
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / num_batches
                ppl = perplexity(avg_loss)
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{ppl:.2f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                if self.config.use_wandb:
                    wandb.log({
                        'train_loss': avg_loss,
                        'train_perplexity': ppl,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
                    
        return {
            'train_loss': total_loss / num_batches,
            'train_perplexity': perplexity(total_loss / num_batches)
        }
        
    def _validate_epoch(self) -> Dict[str, float]:
        """검증 에포크"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 배치를 디바이스로 이동
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = self.loss_fn(outputs, batch)
                else:
                    outputs = self.model(**batch)
                    loss = self.loss_fn(outputs, batch)
                    
                total_loss += loss.item()
                if 'labels' in batch:
                    total_tokens += (batch['labels'] != -100).sum().item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        ppl = perplexity(avg_loss)
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': ppl
        }
        
    def evaluate(self, test_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """모델 평가"""
        if test_loader is None:
            test_loader = self.test_loader
            
        if test_loader is None:
            raise ValueError("No test loader provided")
            
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                # 배치를 디바이스로 이동
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = self.loss_fn(outputs, batch)
                else:
                    outputs = self.model(**batch)
                    loss = self.loss_fn(outputs, batch)
                    
                total_loss += loss.item()
                num_batches += 1
                
                # 예측값 수집
                if isinstance(outputs, torch.Tensor):
                    predictions = torch.argmax(outputs, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    
                if 'labels' in batch:
                    all_labels.extend(batch['labels'].cpu().numpy())
                    
        # 메트릭 계산
        metrics = {
            'test_loss': total_loss / num_batches,
            'test_perplexity': perplexity(total_loss / num_batches)
        }
        
        if all_predictions and all_labels:
            additional_metrics = calculate_metrics(all_predictions, all_labels)
            metrics.update(additional_metrics)
            
        return metrics
        
    def _save_model(self, filename: str):
        """모델 저장"""
        save_path = os.path.join(self.config.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss
        }, save_path)
        
    def _save_training_history(self):
        """훈련 히스토리 저장"""
        history_path = os.path.join(self.config.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
            
    def load_model(self, checkpoint_path: str):
        """모델 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
