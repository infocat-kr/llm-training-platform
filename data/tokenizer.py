"""
토크나이저 구현
"""

import re
import json
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import torch


class SimpleTokenizer:
    """간단한 문자/단어 기반 토크나이저"""
    
    def __init__(self, vocab_size=30000, min_freq=2, special_tokens=None):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab = set()
        
    def build_vocab(self, texts: List[str]):
        """어휘 사전 구축"""
        # 특수 토큰 추가
        for token in self.special_tokens:
            self.word_to_idx[token] = len(self.word_to_idx)
            
        # 단어 빈도 계산
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
            
        # 빈도 기반 어휘 선택
        for word, count in word_counts.most_common():
            if count >= self.min_freq and len(self.word_to_idx) < self.vocab_size:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
                    
        # 역방향 매핑 생성
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab = set(self.word_to_idx.keys())
        
    def _tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분할"""
        # 간단한 정규식 기반 토크나이징
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
        
    def encode(self, text: str) -> List[int]:
        """텍스트를 인덱스로 변환"""
        tokens = self._tokenize(text)
        return [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        
    def decode(self, indices: List[int]) -> str:
        """인덱스를 텍스트로 변환"""
        tokens = [self.idx_to_word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(tokens)
        
    def save(self, filepath: str):
        """토크나이저 저장"""
        data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load(self, filepath: str):
        """토크나이저 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        self.vocab_size = data['vocab_size']
        self.min_freq = data['min_freq']
        self.special_tokens = data['special_tokens']
        self.vocab = set(self.word_to_idx.keys())


class BPETokenizer:
    """Byte Pair Encoding 토크나이저"""
    
    def __init__(self, vocab_size=30000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.bpe_merges = []
        self.vocab = set()
        
    def build_vocab(self, texts: List[str]):
        """BPE 어휘 사전 구축"""
        # 특수 토큰 추가
        for token in self.special_tokens:
            self.word_to_idx[token] = len(self.word_to_idx)
            
        # 문자 단위로 분할
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] += 1
                
        # BPE 알고리즘 적용
        self._apply_bpe(word_counts)
        
        # 역방향 매핑 생성
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab = set(self.word_to_idx.keys())
        
    def _tokenize(self, text: str) -> List[str]:
        """텍스트를 단어로 분할"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
        
    def _apply_bpe(self, word_counts: Counter):
        """BPE 알고리즘 적용"""
        # 초기 어휘 (문자 단위)
        vocab = set()
        for word in word_counts.keys():
            vocab.update(list(word))
            
        # 문자를 어휘에 추가
        for char in vocab:
            if char not in self.word_to_idx:
                self.word_to_idx[char] = len(self.word_to_idx)
                
        # BPE 반복
        while len(self.word_to_idx) < self.vocab_size:
            # 가장 빈번한 쌍 찾기
            pairs = self._get_pairs(word_counts)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            self.bpe_merges.append(best_pair)
            
            # 새로운 토큰 추가
            new_token = ''.join(best_pair)
            self.word_to_idx[new_token] = len(self.word_to_idx)
            
            # 단어 업데이트
            word_counts = self._merge_pair(word_counts, best_pair)
            
    def _get_pairs(self, word_counts: Counter) -> Dict[Tuple[str, str], int]:
        """인접한 토큰 쌍의 빈도 계산"""
        pairs = defaultdict(int)
        for word, count in word_counts.items():
            symbols = self._word_to_bpe_tokens(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += count
        return pairs
        
    def _word_to_bpe_tokens(self, word: str) -> List[str]:
        """단어를 BPE 토큰으로 분할"""
        if word in self.word_to_idx:
            return [word]
        return list(word)
        
    def _merge_pair(self, word_counts: Counter, pair: Tuple[str, str]) -> Counter:
        """특정 쌍을 병합하여 새로운 단어 생성"""
        new_word_counts = Counter()
        for word, count in word_counts.items():
            new_word = word.replace(''.join(pair), ''.join(pair))
            new_word_counts[new_word] = count
        return new_word_counts
        
    def encode(self, text: str) -> List[int]:
        """텍스트를 인덱스로 변환"""
        words = self._tokenize(text)
        tokens = []
        
        for word in words:
            word_tokens = self._apply_bpe_to_word(word)
            tokens.extend([self.word_to_idx.get(token, self.word_to_idx['<UNK>']) 
                          for token in word_tokens])
            
        return tokens
        
    def _apply_bpe_to_word(self, word: str) -> List[str]:
        """단어에 BPE 적용"""
        if word in self.word_to_idx:
            return [word]
            
        symbols = list(word)
        for pair in self.bpe_merges:
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                    new_symbols.append(''.join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols
        
    def decode(self, indices: List[int]) -> str:
        """인덱스를 텍스트로 변환"""
        tokens = [self.idx_to_word.get(idx, '<UNK>') for idx in indices]
        return ''.join(tokens).replace('▁', ' ')
        
    def save(self, filepath: str):
        """토크나이저 저장"""
        data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'bpe_merges': self.bpe_merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def load(self, filepath: str):
        """토크나이저 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        self.bpe_merges = data['bpe_merges']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.vocab = set(self.word_to_idx.keys())


class TokenizerFactory:
    """토크나이저 팩토리 클래스"""
    
    @staticmethod
    def create_tokenizer(tokenizer_type: str, **kwargs):
        """토크나이저 생성"""
        if tokenizer_type == 'simple':
            return SimpleTokenizer(**kwargs)
        elif tokenizer_type == 'bpe':
            return BPETokenizer(**kwargs)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
            
    @staticmethod
    def load_tokenizer(filepath: str, tokenizer_type: str = None):
        """저장된 토크나이저 로드"""
        if tokenizer_type is None:
            # 파일에서 타입 추론
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            tokenizer_type = 'bpe' if 'bpe_merges' in data else 'simple'
            
        tokenizer = TokenizerFactory.create_tokenizer(tokenizer_type)
        tokenizer.load(filepath)
        return tokenizer
