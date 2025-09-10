"""
평가 메트릭들
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
from typing import List, Dict, Any, Optional, Union
import math


def perplexity(loss: float) -> float:
    """Perplexity 계산"""
    return math.exp(loss)


def calculate_metrics(predictions: Union[List, np.ndarray], 
                     labels: Union[List, np.ndarray],
                     average: str = 'weighted') -> Dict[str, float]:
    """분류 메트릭 계산"""
    
    # NumPy 배열로 변환
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    if isinstance(labels, list):
        labels = np.array(labels)
        
    # 기본 메트릭
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # 이진 분류인 경우 AUC 계산
    if len(np.unique(labels)) == 2:
        try:
            # 확률 점수가 필요한 경우 (여기서는 간단히 처리)
            auc = roc_auc_score(labels, predictions)
            metrics['auc'] = auc
        except ValueError:
            pass
            
    return metrics


def bleu_score(predictions: List[List[str]], 
               references: List[List[str]], 
               max_n: int = 4) -> Dict[str, float]:
    """BLEU 점수 계산 (간단한 구현)"""
    
    def get_ngrams(tokens: List[str], n: int) -> Dict[str, int]:
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def precision_n(pred_ngrams: Dict[str, int], 
                   ref_ngrams: Dict[str, int]) -> float:
        if not pred_ngrams:
            return 0.0
            
        overlap = 0
        for ngram in pred_ngrams:
            overlap += min(pred_ngrams[ngram], ref_ngrams.get(ngram, 0))
            
        return overlap / sum(pred_ngrams.values())
    
    # BLEU 점수 계산
    bleu_scores = {}
    precisions = []
    
    for n in range(1, max_n + 1):
        total_precision = 0
        count = 0
        
        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            if pred_ngrams:
                total_precision += precision_n(pred_ngrams, ref_ngrams)
                count += 1
                
        if count > 0:
            avg_precision = total_precision / count
            precisions.append(avg_precision)
            bleu_scores[f'bleu_{n}'] = avg_precision
        else:
            precisions.append(0.0)
            bleu_scores[f'bleu_{n}'] = 0.0
    
    # BLEU-4 계산 (기하평균)
    if precisions:
        bleu_4 = math.exp(sum(math.log(p) for p in precisions if p > 0) / len([p for p in precisions if p > 0]))
        bleu_scores['bleu_4'] = bleu_4
    else:
        bleu_scores['bleu_4'] = 0.0
        
    return bleu_scores


def rouge_score(predictions: List[List[str]], 
                references: List[List[str]], 
                rouge_type: str = 'rouge_l') -> Dict[str, float]:
    """ROUGE 점수 계산 (간단한 구현)"""
    
    def lcs_length(seq1: List[str], seq2: List[str]) -> int:
        """최장 공통 부분 수열 길이"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
    
    if rouge_type == 'rouge_l':
        total_precision = 0
        total_recall = 0
        count = 0
        
        for pred, ref in zip(predictions, references):
            lcs_len = lcs_length(pred, ref)
            
            if len(pred) > 0:
                precision = lcs_len / len(pred)
                total_precision += precision
                
            if len(ref) > 0:
                recall = lcs_len / len(ref)
                total_recall += recall
                
            count += 1
            
        if count > 0:
            avg_precision = total_precision / count
            avg_recall = total_recall / count
            
            if avg_precision + avg_recall > 0:
                f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            else:
                f1 = 0.0
                
            return {
                'rouge_l_precision': avg_precision,
                'rouge_l_recall': avg_recall,
                'rouge_l_f1': f1
            }
        else:
            return {
                'rouge_l_precision': 0.0,
                'rouge_l_recall': 0.0,
                'rouge_l_f1': 0.0
            }
    else:
        raise ValueError(f"Unsupported ROUGE type: {rouge_type}")


def distinct_ngrams(tokens: List[List[str]], n: int = 2) -> float:
    """Distinct-n 점수 계산"""
    all_ngrams = set()
    total_ngrams = 0
    
    for token_list in tokens:
        for i in range(len(token_list) - n + 1):
            ngram = ' '.join(token_list[i:i+n])
            all_ngrams.add(ngram)
            total_ngrams += 1
            
    if total_ngrams == 0:
        return 0.0
        
    return len(all_ngrams) / total_ngrams


def self_bleu_score(tokens: List[List[str]], n: int = 2) -> float:
    """Self-BLEU 점수 계산 (반복성 측정)"""
    if len(tokens) < 2:
        return 0.0
        
    total_bleu = 0
    count = 0
    
    for i, pred in enumerate(tokens):
        references = [tokens[j] for j in range(len(tokens)) if j != i]
        
        if references:
            # 간단한 BLEU 계산
            pred_ngrams = {}
            for j in range(len(pred) - n + 1):
                ngram = ' '.join(pred[j:j+n])
                pred_ngrams[ngram] = pred_ngrams.get(ngram, 0) + 1
                
            max_overlap = 0
            for ref in references:
                ref_ngrams = {}
                for j in range(len(ref) - n + 1):
                    ngram = ' '.join(ref[j:j+n])
                    ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
                    
                overlap = 0
                for ngram in pred_ngrams:
                    overlap += min(pred_ngrams[ngram], ref_ngrams.get(ngram, 0))
                    
                max_overlap = max(max_overlap, overlap)
                
            if sum(pred_ngrams.values()) > 0:
                bleu = max_overlap / sum(pred_ngrams.values())
                total_bleu += bleu
                count += 1
                
    return total_bleu / count if count > 0 else 0.0


def calculate_generation_metrics(predictions: List[List[str]], 
                                references: List[List[str]]) -> Dict[str, float]:
    """텍스트 생성 메트릭 계산"""
    metrics = {}
    
    # BLEU 점수
    bleu_scores = bleu_score(predictions, references)
    metrics.update(bleu_scores)
    
    # ROUGE 점수
    rouge_scores = rouge_score(predictions, references)
    metrics.update(rouge_scores)
    
    # Distinct 점수
    for n in [1, 2, 3, 4]:
        distinct_n = distinct_ngrams(predictions, n)
        metrics[f'distinct_{n}'] = distinct_n
        
    # Self-BLEU 점수
    for n in [2, 3, 4]:
        self_bleu_n = self_bleu_score(predictions, n)
        metrics[f'self_bleu_{n}'] = self_bleu_n
        
    return metrics


def calculate_language_model_metrics(logits: torch.Tensor, 
                                   labels: torch.Tensor,
                                   ignore_index: int = -100) -> Dict[str, float]:
    """언어 모델 메트릭 계산"""
    
    # 손실 계산
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                          labels.view(-1), 
                          ignore_index=ignore_index)
    
    # Perplexity
    ppl = perplexity(loss.item())
    
    # 정확도 (유효한 토큰만)
    valid_mask = labels != ignore_index
    if valid_mask.sum() > 0:
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()
    else:
        accuracy = 0.0
        
    return {
        'loss': loss.item(),
        'perplexity': ppl,
        'accuracy': accuracy.item()
    }


def calculate_embedding_similarity(embeddings1: torch.Tensor, 
                                 embeddings2: torch.Tensor,
                                 metric: str = 'cosine') -> torch.Tensor:
    """임베딩 유사도 계산"""
    
    if metric == 'cosine':
        # 코사인 유사도
        embeddings1_norm = F.normalize(embeddings1, p=2, dim=-1)
        embeddings2_norm = F.normalize(embeddings2, p=2, dim=-1)
        similarity = torch.matmul(embeddings1_norm, embeddings2_norm.transpose(-2, -1))
        
    elif metric == 'euclidean':
        # 유클리드 거리 (유사도로 변환)
        distances = torch.cdist(embeddings1, embeddings2, p=2)
        similarity = 1 / (1 + distances)
        
    elif metric == 'dot':
        # 내적
        similarity = torch.matmul(embeddings1, embeddings2.transpose(-2, -1))
        
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")
        
    return similarity


class MetricsCalculator:
    """메트릭 계산기 클래스"""
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        
    def calculate(self, predictions: Any, labels: Any, **kwargs) -> Dict[str, float]:
        """작업 유형에 따른 메트릭 계산"""
        
        if self.task_type == 'classification':
            return calculate_metrics(predictions, labels, **kwargs)
        elif self.task_type == 'generation':
            return calculate_generation_metrics(predictions, labels)
        elif self.task_type == 'language_model':
            return calculate_language_model_metrics(predictions, labels, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
