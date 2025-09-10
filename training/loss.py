"""
손실 함수들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class LanguageModelLoss(nn.Module):
    """언어 모델 손실 함수 (Cross Entropy)"""
    
    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
    def forward(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            outputs: (batch_size, seq_len, vocab_size)
            batch: 배치 데이터 (labels 포함)
        """
        if 'labels' in batch:
            labels = batch['labels']
        else:
            # GPT 스타일: 입력을 한 칸씩 시프트한 것이 라벨
            labels = batch['input_ids'][:, 1:]
            outputs = outputs[:, :-1, :]
            
        # Cross entropy loss
        loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )
        
        return loss


class ClassificationLoss(nn.Module):
    """분류 손실 함수"""
    
    def __init__(self, num_classes: int, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
    def forward(self, outputs: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            outputs: (batch_size, num_classes)
            batch: 배치 데이터 (labels 포함)
        """
        labels = batch['labels']
        
        loss = F.cross_entropy(
            outputs,
            labels,
            label_smoothing=self.label_smoothing
        )
        
        return loss


class ContrastiveLoss(nn.Module):
    """대조 학습 손실 함수"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: (batch_size, hidden_size)
            positive: (batch_size, hidden_size)
            negative: (batch_size, num_negatives, hidden_size)
        """
        # 정규화
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=2)
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        # Negative similarities
        neg_sim = torch.bmm(negative, anchor.unsqueeze(2)).squeeze(2) / self.temperature
        
        # Logits
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss


class FocalLoss(nn.Module):
    """Focal Loss (불균형 데이터셋용)"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes)
            targets: (batch_size,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """라벨 스무딩이 적용된 Cross Entropy"""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes)
            targets: (batch_size,)
        """
        log_preds = F.log_softmax(inputs, dim=1)
        nll_loss = -log_preds.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class KLDivLoss(nn.Module):
    """KL Divergence Loss (지식 증류용)"""
    
    def __init__(self, temperature: float = 3.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits: (batch_size, num_classes)
            teacher_logits: (batch_size, num_classes)
        """
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        loss = F.kl_div(student_probs, teacher_probs, reduction='none')
        loss = loss * (self.temperature ** 2)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """다중 작업 손실 함수"""
    
    def __init__(self, task_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.task_weights = task_weights or {}
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            outputs: 각 작업별 모델 출력
            targets: 각 작업별 타겟
        """
        total_loss = 0.0
        
        for task_name, output in outputs.items():
            target = targets[task_name]
            weight = self.task_weights.get(task_name, 1.0)
            
            if task_name == 'classification':
                loss = F.cross_entropy(output, target)
            elif task_name == 'regression':
                loss = F.mse_loss(output, target)
            elif task_name == 'language_model':
                loss = F.cross_entropy(
                    output.reshape(-1, output.size(-1)),
                    target.reshape(-1),
                    ignore_index=-100
                )
            else:
                raise ValueError(f"Unknown task: {task_name}")
                
            total_loss += weight * loss
            
        return total_loss


def create_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """손실 함수 생성 팩토리"""
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'language_model':
        return LanguageModelLoss(**kwargs)
    elif loss_type == 'classification':
        return ClassificationLoss(**kwargs)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    elif loss_type == 'kl_div':
        return KLDivLoss(**kwargs)
    elif loss_type == 'multi_task':
        return MultiTaskLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
