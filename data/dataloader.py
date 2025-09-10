"""
데이터로더 유틸리티
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional, Dict, Any
import random
import numpy as np


def create_dataloader(dataset, 
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     pin_memory: bool = True,
                     drop_last: bool = False) -> DataLoader:
    """데이터로더 생성"""
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """배치 데이터 정렬 함수"""
    # 배치에서 각 필드 추출
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    # 라벨이 있으면 추가
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        result['labels'] = labels
        
    # 텍스트 데이터가 있으면 추가
    if 'text' in batch[0]:
        result['texts'] = [item['text'] for item in batch]
        
    if 'question' in batch[0]:
        result['questions'] = [item['question'] for item in batch]
        result['answers'] = [item['answer'] for item in batch]
        
    if 'conversation' in batch[0]:
        result['conversations'] = [item['conversation'] for item in batch]
        
    return result


def split_dataset(dataset, 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_seed: int = 42) -> Tuple[torch.utils.data.Dataset, 
                                               torch.utils.data.Dataset, 
                                               torch.utils.data.Dataset]:
    """데이터셋을 훈련/검증/테스트로 분할"""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    return random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )


def create_dataloaders(dataset,
                      batch_size: int = 32,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      num_workers: int = 0,
                      pin_memory: bool = True,
                      random_seed: int = 42) -> Dict[str, DataLoader]:
    """훈련/검증/테스트 데이터로더 생성"""
    
    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # 데이터로더 생성
    train_loader = create_dataloader(
        train_dataset, batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = create_dataloader(
        val_dataset, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = create_dataloader(
        test_dataset, batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


class InfiniteDataLoader:
    """무한 반복 데이터로더"""
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


def set_seed(seed: int = 42):
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataLoaderConfig:
    """데이터로더 설정 클래스"""
    
    def __init__(self,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 shuffle_train: bool = True,
                 shuffle_val: bool = False,
                 shuffle_test: bool = False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        
    def create_dataloader(self, dataset, split: str = 'train') -> DataLoader:
        """설정에 따라 데이터로더 생성"""
        shuffle = getattr(self, f'shuffle_{split}', False)
        
        return create_dataloader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
