"""
텍스트 생성기
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Union
import time
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """텍스트 생성 설정"""
    max_length: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = True
    do_sample: bool = True
    num_beams: int = 1
    num_return_sequences: int = 1
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1
    unk_token_id: int = 3


class TextGenerator:
    """텍스트 생성기"""
    
    def __init__(self, model, tokenizer, config: GenerationConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
    def generate(self, 
                input_text: str,
                config: Optional[GenerationConfig] = None) -> List[str]:
        """텍스트 생성"""
        
        config = config or self.config
        
        # 입력 토크나이징
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self.model = self.model.cuda()
            
        # 생성
        if config.num_beams > 1:
            outputs = self._beam_search(input_tensor, config)
        else:
            outputs = self._greedy_search(input_tensor, config)
            
        # 디코딩
        generated_texts = []
        for output in outputs:
            generated_tokens = output[len(input_ids):]
            generated_text = self.tokenizer.decode(generated_tokens)
            generated_texts.append(generated_text)
            
        return generated_texts
        
    def _greedy_search(self, input_ids: torch.Tensor, config: GenerationConfig) -> List[torch.Tensor]:
        """Greedy 검색"""
        with torch.no_grad():
            current_ids = input_ids.clone()
            generated_sequences = []
            
            for _ in range(config.num_return_sequences):
                sequence = current_ids.clone()
                
                for _ in range(config.max_length):
                    # 모델 예측
                    outputs = self.model(sequence)
                    next_token_logits = outputs[0, -1, :]
                    
                    # 반복 페널티 적용
                    if config.repetition_penalty != 1.0:
                        next_token_logits = self._apply_repetition_penalty(
                            next_token_logits, sequence, config.repetition_penalty
                        )
                    
                    # 샘플링
                    if config.do_sample:
                        next_token = self._sample_token(next_token_logits, config)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1)
                    
                    # 시퀀스에 추가
                    sequence = torch.cat([sequence, next_token.unsqueeze(0)], dim=1)
                    
                    # EOS 토큰이면 중단
                    if next_token.item() == config.eos_token_id:
                        break
                        
                generated_sequences.append(sequence)
                
        return generated_sequences
        
    def _beam_search(self, input_ids: torch.Tensor, config: GenerationConfig) -> List[torch.Tensor]:
        """Beam 검색"""
        with torch.no_grad():
            # 초기 beam
            beams = [(input_ids, 0.0)]  # (sequence, score)
            completed_sequences = []
            
            for step in range(config.max_length):
                new_beams = []
                
                for sequence, score in beams:
                    # 모델 예측
                    outputs = self.model(sequence)
                    next_token_logits = outputs[0, -1, :]
                    
                    # 반복 페널티 적용
                    if config.repetition_penalty != 1.0:
                        next_token_logits = self._apply_repetition_penalty(
                            next_token_logits, sequence, config.repetition_penalty
                        )
                    
                    # Top-k 또는 Top-p 필터링
                    if config.top_k is not None:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, config.top_k)
                        next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                        next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                    
                    if config.top_p is not None:
                        next_token_logits = self._apply_top_p(next_token_logits, config.top_p)
                    
                    # 확률 계산
                    probs = F.softmax(next_token_logits / config.temperature, dim=-1)
                    
                    # 상위 beam_size개 토큰 선택
                    top_probs, top_indices = torch.topk(probs, config.num_beams)
                    
                    for prob, token_id in zip(top_probs, top_indices):
                        new_sequence = torch.cat([sequence, token_id.unsqueeze(0)], dim=1)
                        new_score = score + torch.log(prob)
                        
                        if token_id.item() == config.eos_token_id:
                            completed_sequences.append((new_sequence, new_score))
                        else:
                            new_beams.append((new_sequence, new_score))
                
                # 상위 beam_size개 선택
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:config.num_beams]
                
                if not beams:
                    break
                    
            # 완료된 시퀀스와 현재 beam 합치기
            all_sequences = completed_sequences + beams
            all_sequences.sort(key=lambda x: x[1], reverse=True)
            
            return [seq for seq, _ in all_sequences[:config.num_return_sequences]]
            
    def _sample_token(self, logits: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """토큰 샘플링"""
        # Top-k 필터링
        if config.top_k is not None:
            top_k_logits, top_k_indices = torch.topk(logits, config.top_k)
            filtered_logits = torch.full_like(logits, -float('inf'))
            filtered_logits.scatter_(0, top_k_indices, top_k_logits)
            logits = filtered_logits
            
        # Top-p 필터링
        if config.top_p is not None:
            logits = self._apply_top_p(logits, config.top_p)
            
        # 온도 적용
        if config.temperature != 1.0:
            logits = logits / config.temperature
            
        # 샘플링
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        
        return token
        
    def _apply_repetition_penalty(self, logits: torch.Tensor, sequence: torch.Tensor, 
                                penalty: float) -> torch.Tensor:
        """반복 페널티 적용"""
        for token_id in sequence[0]:
            if logits[token_id] < 0:
                logits[token_id] *= penalty
            else:
                logits[token_id] /= penalty
        return logits
        
    def _apply_top_p(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Top-p (nucleus) 필터링"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 누적 확률이 top_p를 초과하는 지점 찾기
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 해당 토큰들의 로짓을 -inf로 설정
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        
        return logits
        
    def generate_batch(self, 
                      input_texts: List[str],
                      config: Optional[GenerationConfig] = None) -> List[List[str]]:
        """배치 텍스트 생성"""
        results = []
        for text in input_texts:
            generated = self.generate(text, config)
            results.append(generated)
        return results
        
    def chat(self, 
             message: str,
             history: Optional[List[Dict[str, str]]] = None,
             config: Optional[GenerationConfig] = None) -> str:
        """채팅 응답 생성"""
        
        if history is None:
            history = []
            
        # 대화 히스토리를 하나의 프롬프트로 결합
        prompt = self._format_chat_prompt(message, history)
        
        # 텍스트 생성
        generated_texts = self.generate(prompt, config)
        response = generated_texts[0] if generated_texts else ""
        
        return response
        
    def _format_chat_prompt(self, message: str, history: List[Dict[str, str]]) -> str:
        """채팅 프롬프트 포맷팅"""
        prompt = ""
        
        for turn in history:
            prompt += f"Human: {turn['human']}\nAssistant: {turn['assistant']}\n"
            
        prompt += f"Human: {message}\nAssistant:"
        
        return prompt


class StreamingGenerator(TextGenerator):
    """스트리밍 텍스트 생성기"""
    
    def __init__(self, model, tokenizer, config: GenerationConfig = None):
        super().__init__(model, tokenizer, config)
        
    def generate_stream(self, 
                       input_text: str,
                       config: Optional[GenerationConfig] = None,
                       callback: Optional[callable] = None):
        """스트리밍 텍스트 생성"""
        
        config = config or self.config
        
        # 입력 토크나이징
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        with torch.no_grad():
            current_ids = input_tensor.clone()
            generated_text = ""
            
            for _ in range(config.max_length):
                # 모델 예측
                outputs = self.model(current_ids)
                next_token_logits = outputs[0, -1, :]
                
                # 샘플링
                if config.do_sample:
                    next_token = self._sample_token(next_token_logits, config)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                
                # 토큰을 텍스트로 변환
                token_text = self.tokenizer.decode([next_token.item()])
                generated_text += token_text
                
                # 콜백 호출
                if callback:
                    callback(token_text, generated_text)
                
                # 시퀀스에 추가
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                
                # EOS 토큰이면 중단
                if next_token.item() == config.eos_token_id:
                    break
                    
        return generated_text
