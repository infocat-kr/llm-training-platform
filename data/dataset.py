"""
데이터셋 클래스들
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
import json
import os
from .tokenizer import SimpleTokenizer, BPETokenizer
from .preprocessing import TextPreprocessor


class TextDataset(Dataset):
    """텍스트 데이터셋"""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer,
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        
        # 전처리 적용
        if self.preprocessor:
            self.texts = self.preprocessor.preprocess_batch(self.texts)
            
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 토크나이징
        tokens = self.tokenizer.encode(text)
        
        # 길이 제한
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        # 패딩
        attention_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.word_to_idx.get('<PAD>', 0))
            attention_mask.append(0)
            
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'text': text
        }


class LanguageModelDataset(Dataset):
    """언어 모델 훈련용 데이터셋 (GPT 스타일)"""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer,
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        
        # 전처리 적용
        if self.preprocessor:
            self.texts = self.preprocessor.preprocess_batch(self.texts)
            
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # BOS, EOS 토큰 추가
        tokens = [self.tokenizer.word_to_idx.get('<BOS>', 1)]
        tokens.extend(self.tokenizer.encode(text))
        tokens.append(self.tokenizer.word_to_idx.get('<EOS>', 2))
        
        # 길이 제한
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        # 입력과 타겟 생성 (shifted by 1)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # 패딩
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.word_to_idx.get('<PAD>', 0))
            labels.append(-100)  # CrossEntropyLoss에서 무시
            attention_mask.append(0)
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'text': text
        }


class ConversationDataset(Dataset):
    """대화 데이터셋"""
    
    def __init__(self, 
                 conversations: List[Dict[str, Any]],
                 tokenizer,
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.conversations)
        
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # 대화를 하나의 텍스트로 결합
        if isinstance(conversation, dict):
            if 'messages' in conversation:
                text = ' '.join([msg.get('content', '') for msg in conversation['messages']])
            elif 'conversation' in conversation:
                text = conversation['conversation']
            else:
                text = str(conversation)
        else:
            text = str(conversation)
            
        # 전처리
        if self.preprocessor:
            text = self.preprocessor.preprocess(text)
            
        # 토크나이징
        tokens = self.tokenizer.encode(text)
        
        # 길이 제한
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            
        # 패딩
        attention_mask = [1] * len(tokens)
        while len(tokens) < self.max_length:
            tokens.append(self.tokenizer.word_to_idx.get('<PAD>', 0))
            attention_mask.append(0)
            
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'conversation': conversation
        }


class QADataset(Dataset):
    """질문-답변 데이터셋"""
    
    def __init__(self, 
                 qa_pairs: List[Tuple[str, str]],
                 tokenizer,
                 max_length: int = 512,
                 preprocessor: Optional[TextPreprocessor] = None):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.qa_pairs)
        
    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        
        # 전처리
        if self.preprocessor:
            question = self.preprocessor.preprocess(question)
            answer = self.preprocessor.preprocess(answer)
            
        # 질문 토크나이징
        question_tokens = self.tokenizer.encode(question)
        answer_tokens = self.tokenizer.encode(answer)
        
        # 길이 제한
        total_length = len(question_tokens) + len(answer_tokens) + 2  # +2 for special tokens
        if total_length > self.max_length:
            # 질문을 우선적으로 유지
            available_length = self.max_length - len(answer_tokens) - 2
            if available_length > 0:
                question_tokens = question_tokens[:available_length]
            else:
                question_tokens = question_tokens[:self.max_length//2]
                answer_tokens = answer_tokens[:self.max_length//2]
                
        # 입력 생성 (질문 + 답변)
        input_tokens = question_tokens + answer_tokens
        
        # 패딩
        attention_mask = [1] * len(input_tokens)
        while len(input_tokens) < self.max_length:
            input_tokens.append(self.tokenizer.word_to_idx.get('<PAD>', 0))
            attention_mask.append(0)
            
        return {
            'input_ids': torch.tensor(input_tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'question': question,
            'answer': answer
        }


class DatasetFactory:
    """데이터셋 팩토리 클래스"""
    
    @staticmethod
    def create_dataset(dataset_type: str, data, tokenizer, **kwargs):
        """데이터셋 생성"""
        if dataset_type == 'text':
            return TextDataset(data, tokenizer, **kwargs)
        elif dataset_type == 'language_model':
            return LanguageModelDataset(data, tokenizer, **kwargs)
        elif dataset_type == 'conversation':
            return ConversationDataset(data, tokenizer, **kwargs)
        elif dataset_type == 'qa':
            return QADataset(data, tokenizer, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
    @staticmethod
    def load_from_file(filepath: str, dataset_type: str, tokenizer, **kwargs):
        """파일에서 데이터셋 로드"""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = f.readlines()
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
            
        return DatasetFactory.create_dataset(dataset_type, data, tokenizer, **kwargs)
