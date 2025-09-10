"""
모델 평가기
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .generator import TextGenerator, GenerationConfig


class ModelEvaluator:
    """모델 평가기"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = TextGenerator(model, tokenizer)
        
    def evaluate_language_model(self, 
                               test_data: List[Dict[str, Any]],
                               generation_config: Optional[GenerationConfig] = None) -> Dict[str, float]:
        """언어 모델 평가"""
        
        total_loss = 0
        total_tokens = 0
        perplexities = []
        
        self.model.eval()
        
        with torch.no_grad():
            for item in test_data:
                if 'input_ids' in item and 'labels' in item:
                    # 이미 토크나이징된 데이터
                    input_ids = torch.tensor(item['input_ids']).unsqueeze(0)
                    labels = torch.tensor(item['labels']).unsqueeze(0)
                else:
                    # 텍스트 데이터
                    text = item.get('text', '')
                    input_ids = self.tokenizer.encode(text)
                    input_tensor = torch.tensor([input_ids], dtype=torch.long)
                    
                    # 라벨 생성 (한 칸 시프트)
                    labels = torch.cat([input_tensor[:, 1:], 
                                      torch.tensor([[self.tokenizer.word_to_idx.get('<EOS>', 2)]])], dim=1)
                    input_ids = input_tensor
                
                # 모델 예측
                outputs = self.model(input_ids)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                valid_tokens = (labels != -100).sum().item()
                total_tokens += valid_tokens
                
                # Perplexity 계산
                ppl = torch.exp(loss).item()
                perplexities.append(ppl)
                
        avg_loss = total_loss / len(test_data)
        avg_perplexity = np.mean(perplexities)
        
        return {
            'loss': avg_loss,
            'perplexity': avg_perplexity,
            'total_tokens': total_tokens
        }
        
    def evaluate_generation(self, 
                           test_data: List[Dict[str, Any]],
                           generation_config: Optional[GenerationConfig] = None) -> Dict[str, float]:
        """텍스트 생성 평가"""
        
        predictions = []
        references = []
        generation_times = []
        
        for item in test_data:
            input_text = item.get('input', item.get('prompt', ''))
            reference = item.get('output', item.get('target', ''))
            
            # 생성 시간 측정
            start_time = time.time()
            generated_texts = self.generator.generate(input_text, generation_config)
            generation_time = time.time() - start_time
            
            predictions.append(generated_texts[0] if generated_texts else "")
            references.append(reference)
            generation_times.append(generation_time)
            
        # BLEU 점수 계산
        bleu_scores = self._calculate_bleu(predictions, references)
        
        # ROUGE 점수 계산
        rouge_scores = self._calculate_rouge(predictions, references)
        
        # Distinct 점수 계산
        distinct_scores = self._calculate_distinct(predictions)
        
        # Self-BLEU 점수 계산
        self_bleu_scores = self._calculate_self_bleu(predictions)
        
        return {
            **bleu_scores,
            **rouge_scores,
            **distinct_scores,
            **self_bleu_scores,
            'avg_generation_time': np.mean(generation_times),
            'total_samples': len(test_data)
        }
        
    def evaluate_classification(self, 
                               test_data: List[Dict[str, Any]],
                               labels: List[str]) -> Dict[str, float]:
        """분류 작업 평가"""
        
        predictions = []
        true_labels = []
        
        for item in test_data:
            text = item.get('text', '')
            true_label = item.get('label', '')
            
            # 각 라벨에 대한 점수 계산
            label_scores = []
            for label in labels:
                prompt = f"Text: {text}\nLabel: {label}\nScore:"
                generated = self.generator.generate(prompt)[0]
                
                # 점수 추출 (간단한 파싱)
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', generated)
                    score = float(numbers[0]) if numbers else 0.5
                    if score > 1:
                        score = score / 100
                except:
                    score = 0.5
                    
                label_scores.append(score)
                
            # 예측 라벨
            predicted_label = labels[np.argmax(label_scores)]
            predictions.append(predicted_label)
            true_labels.append(true_label)
            
        # 메트릭 계산
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    def evaluate_qa(self, 
                   test_data: List[Dict[str, Any]],
                   generation_config: Optional[GenerationConfig] = None) -> Dict[str, float]:
        """질문-답변 평가"""
        
        exact_matches = 0
        f1_scores = []
        generation_times = []
        
        for item in test_data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            # 답변 생성
            start_time = time.time()
            generated_texts = self.generator.generate(question, generation_config)
            generation_time = time.time() - start_time
            
            generated_answer = generated_texts[0] if generated_texts else ""
            generation_times.append(generation_time)
            
            # Exact Match
            if generated_answer.strip().lower() == answer.strip().lower():
                exact_matches += 1
                
            # F1 Score
            f1 = self._calculate_f1(generated_answer, answer)
            f1_scores.append(f1)
            
        return {
            'exact_match': exact_matches / len(test_data),
            'f1_score': np.mean(f1_scores),
            'avg_generation_time': np.mean(generation_times),
            'total_samples': len(test_data)
        }
        
    def _calculate_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """BLEU 점수 계산"""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        smoothing = SmoothingFunction().method1
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if pred_tokens and ref_tokens:
                bleu_1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
                bleu_2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                bleu_3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                bleu_4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                
                bleu_1_scores.append(bleu_1)
                bleu_2_scores.append(bleu_2)
                bleu_3_scores.append(bleu_3)
                bleu_4_scores.append(bleu_4)
                
        return {
            'bleu_1': np.mean(bleu_1_scores) if bleu_1_scores else 0.0,
            'bleu_2': np.mean(bleu_2_scores) if bleu_2_scores else 0.0,
            'bleu_3': np.mean(bleu_3_scores) if bleu_3_scores else 0.0,
            'bleu_4': np.mean(bleu_4_scores) if bleu_4_scores else 0.0
        }
        
    def _calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """ROUGE 점수 계산 (간단한 구현)"""
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if pred_tokens and ref_tokens:
                # ROUGE-1
                pred_1grams = set(pred_tokens)
                ref_1grams = set(ref_tokens)
                overlap_1 = len(pred_1grams & ref_1grams)
                rouge_1 = overlap_1 / len(ref_1grams) if ref_1grams else 0
                rouge_1_scores.append(rouge_1)
                
                # ROUGE-2
                pred_2grams = set(zip(pred_tokens, pred_tokens[1:]))
                ref_2grams = set(zip(ref_tokens, ref_tokens[1:]))
                overlap_2 = len(pred_2grams & ref_2grams)
                rouge_2 = overlap_2 / len(ref_2grams) if ref_2grams else 0
                rouge_2_scores.append(rouge_2)
                
                # ROUGE-L (LCS)
                lcs_length = self._lcs_length(pred_tokens, ref_tokens)
                rouge_l = lcs_length / len(ref_tokens) if ref_tokens else 0
                rouge_l_scores.append(rouge_l)
                
        return {
            'rouge_1': np.mean(rouge_1_scores) if rouge_1_scores else 0.0,
            'rouge_2': np.mean(rouge_2_scores) if rouge_2_scores else 0.0,
            'rouge_l': np.mean(rouge_l_scores) if rouge_l_scores else 0.0
        }
        
    def _calculate_distinct(self, predictions: List[str]) -> Dict[str, float]:
        """Distinct 점수 계산"""
        distinct_1_scores = []
        distinct_2_scores = []
        
        for pred in predictions:
            tokens = pred.split()
            
            if len(tokens) >= 1:
                # Distinct-1
                unique_1grams = len(set(tokens))
                distinct_1 = unique_1grams / len(tokens)
                distinct_1_scores.append(distinct_1)
                
            if len(tokens) >= 2:
                # Distinct-2
                bigrams = list(zip(tokens, tokens[1:]))
                unique_2grams = len(set(bigrams))
                distinct_2 = unique_2grams / len(bigrams)
                distinct_2_scores.append(distinct_2)
                
        return {
            'distinct_1': np.mean(distinct_1_scores) if distinct_1_scores else 0.0,
            'distinct_2': np.mean(distinct_2_scores) if distinct_2_scores else 0.0
        }
        
    def _calculate_self_bleu(self, predictions: List[str]) -> Dict[str, float]:
        """Self-BLEU 점수 계산"""
        if len(predictions) < 2:
            return {'self_bleu_2': 0.0, 'self_bleu_3': 0.0, 'self_bleu_4': 0.0}
            
        self_bleu_2_scores = []
        self_bleu_3_scores = []
        self_bleu_4_scores = []
        
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoothing = SmoothingFunction().method1
        
        for i, pred in enumerate(predictions):
            pred_tokens = pred.split()
            references = [p.split() for j, p in enumerate(predictions) if j != i and p.split()]
            
            if pred_tokens and references:
                self_bleu_2 = sentence_bleu(references, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
                self_bleu_3 = sentence_bleu(references, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
                self_bleu_4 = sentence_bleu(references, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
                
                self_bleu_2_scores.append(self_bleu_2)
                self_bleu_3_scores.append(self_bleu_3)
                self_bleu_4_scores.append(self_bleu_4)
                
        return {
            'self_bleu_2': np.mean(self_bleu_2_scores) if self_bleu_2_scores else 0.0,
            'self_bleu_3': np.mean(self_bleu_3_scores) if self_bleu_3_scores else 0.0,
            'self_bleu_4': np.mean(self_bleu_4_scores) if self_bleu_4_scores else 0.0
        }
        
    def _calculate_f1(self, prediction: str, reference: str) -> float:
        """F1 점수 계산"""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return 1.0 if not pred_tokens else 0.0
            
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
        
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
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
        
    def benchmark_inference(self, 
                           test_data: List[Dict[str, Any]],
                           num_runs: int = 3) -> Dict[str, float]:
        """추론 성능 벤치마크"""
        
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            for item in test_data:
                text = item.get('text', item.get('input', ''))
                self.generator.generate(text)
                
            end_time = time.time()
            times.append(end_time - start_time)
            
            # 메모리 사용량 (GPU)
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024**3)
                
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times),
            'avg_memory_gb': np.mean(memory_usage) if memory_usage else 0.0,
            'samples_per_second': len(test_data) / np.mean(times)
        }
