"""
추론 엔진
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Union
import time
import json
from dataclasses import dataclass
from .generator import TextGenerator, GenerationConfig


@dataclass
class InferenceConfig:
    """추론 설정"""
    device: str = 'auto'
    batch_size: int = 1
    max_memory_usage: float = 0.8  # GPU 메모리 사용률 제한
    use_half_precision: bool = False
    enable_caching: bool = True
    cache_size: int = 1000


class InferenceEngine:
    """추론 엔진"""
    
    def __init__(self, model, tokenizer, config: InferenceConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()
        
        # 디바이스 설정
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
            
        # 모델을 디바이스로 이동
        self.model.to(self.device)
        
        # Half precision 설정
        if self.config.use_half_precision and self.device.type == 'cuda':
            self.model = self.model.half()
            
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 캐시 초기화
        if self.config.enable_caching:
            self.cache = {}
            
        # 텍스트 생성기 초기화
        self.generator = TextGenerator(model, tokenizer)
        
    def predict(self, 
                input_text: str,
                generation_config: Optional[GenerationConfig] = None) -> Dict[str, Any]:
        """텍스트 예측"""
        
        start_time = time.time()
        
        # 캐시 확인
        cache_key = None
        if self.config.enable_caching:
            cache_key = self._get_cache_key(input_text, generation_config)
            if cache_key in self.cache:
                result = self.cache[cache_key].copy()
                result['from_cache'] = True
                return result
                
        # 텍스트 생성
        generated_texts = self.generator.generate(input_text, generation_config)
        
        # 결과 구성
        result = {
            'input_text': input_text,
            'generated_texts': generated_texts,
            'generation_time': time.time() - start_time,
            'from_cache': False
        }
        
        # 캐시 저장
        if self.config.enable_caching and cache_key:
            self._update_cache(cache_key, result)
            
        return result
        
    def predict_batch(self, 
                     input_texts: List[str],
                     generation_config: Optional[GenerationConfig] = None) -> List[Dict[str, Any]]:
        """배치 예측"""
        
        results = []
        
        for i in range(0, len(input_texts), self.config.batch_size):
            batch_texts = input_texts[i:i + self.config.batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict(text, generation_config)
                batch_results.append(result)
                
            results.extend(batch_results)
            
        return results
        
    def get_embeddings(self, 
                      input_texts: List[str],
                      layer: Optional[int] = None) -> torch.Tensor:
        """텍스트 임베딩 추출"""
        
        embeddings = []
        
        with torch.no_grad():
            for text in input_texts:
                # 토크나이징
                input_ids = self.tokenizer.encode(text)
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                
                # 모델 통과
                if hasattr(self.model, 'get_embeddings'):
                    embedding = self.model.get_embeddings(input_tensor, layer)
                else:
                    # 기본적으로 마지막 hidden state 사용
                    outputs = self.model(input_tensor)
                    if isinstance(outputs, tuple):
                        embedding = outputs[0]  # (hidden_states, ...)
                    else:
                        embedding = outputs
                        
                    # 평균 풀링
                    embedding = embedding.mean(dim=1)
                    
                embeddings.append(embedding.cpu())
                
        return torch.cat(embeddings, dim=0)
        
    def get_attention_weights(self, 
                            input_text: str,
                            layer: Optional[int] = None,
                            head: Optional[int] = None) -> torch.Tensor:
        """어텐션 가중치 추출"""
        
        # 토크나이징
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # 어텐션 가중치 추출 (모델에 따라 구현이 다를 수 있음)
            if hasattr(self.model, 'get_attention_weights'):
                attention_weights = self.model.get_attention_weights(
                    input_tensor, layer, head
                )
            else:
                # 기본 구현 (실제로는 모델에 따라 다름)
                attention_weights = torch.ones(
                    input_tensor.size(1), input_tensor.size(1)
                )
                
        return attention_weights
        
    def compute_similarity(self, 
                          text1: str, 
                          text2: str,
                          metric: str = 'cosine') -> float:
        """텍스트 유사도 계산"""
        
        # 임베딩 추출
        embeddings = self.get_embeddings([text1, text2])
        emb1, emb2 = embeddings[0], embeddings[1]
        
        # 유사도 계산
        if metric == 'cosine':
            similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
        elif metric == 'euclidean':
            distance = torch.norm(emb1 - emb2)
            similarity = 1 / (1 + distance)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
            
        return similarity.item()
        
    def classify_text(self, 
                     input_text: str,
                     labels: List[str],
                     return_probabilities: bool = True) -> Dict[str, Any]:
        """텍스트 분류"""
        
        # 각 라벨에 대한 확률 계산
        label_scores = []
        
        for label in labels:
            # 프롬프트 구성 (간단한 예시)
            prompt = f"Text: {input_text}\nLabel: {label}\nProbability:"
            
            # 생성
            result = self.predict(prompt)
            generated_text = result['generated_texts'][0]
            
            # 확률 추출 (간단한 파싱)
            try:
                # 생성된 텍스트에서 숫자 추출
                import re
                numbers = re.findall(r'\d+\.?\d*', generated_text)
                if numbers:
                    score = float(numbers[0])
                    if score > 1:
                        score = score / 100  # 퍼센트를 확률로 변환
                else:
                    score = 0.5  # 기본값
            except:
                score = 0.5
                
            label_scores.append(score)
            
        # 정규화
        total_score = sum(label_scores)
        if total_score > 0:
            probabilities = [score / total_score for score in label_scores]
        else:
            probabilities = [1.0 / len(labels)] * len(labels)
            
        # 결과
        result = {
            'input_text': input_text,
            'labels': labels,
            'probabilities': probabilities,
            'predicted_label': labels[probabilities.index(max(probabilities))]
        }
        
        return result
        
    def _get_cache_key(self, input_text: str, config: Optional[GenerationConfig]) -> str:
        """캐시 키 생성"""
        config_str = json.dumps(config.__dict__ if config else {}, sort_keys=True)
        return f"{input_text}_{config_str}"
        
    def _update_cache(self, cache_key: str, result: Dict[str, Any]):
        """캐시 업데이트"""
        if len(self.cache) >= self.config.cache_size:
            # 가장 오래된 항목 제거
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            
        self.cache[cache_key] = result
        
    def clear_cache(self):
        """캐시 초기화"""
        if hasattr(self, 'cache'):
            self.cache.clear()
            
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_type': type(self.model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'half_precision': self.config.use_half_precision,
            'cache_enabled': self.config.enable_caching,
            'cache_size': len(self.cache) if hasattr(self, 'cache') else 0
        }
        
        if torch.cuda.is_available():
            info['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
            info['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
            
        return info
        
    def benchmark(self, 
                 input_texts: List[str],
                 num_runs: int = 5) -> Dict[str, float]:
        """성능 벤치마크"""
        
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            self.predict_batch(input_texts)
            end_time = time.time()
            times.append(end_time - start_time)
            
        return {
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            'tokens_per_second': sum(len(self.tokenizer.encode(text)) for text in input_texts) / (sum(times) / len(times))
        }
