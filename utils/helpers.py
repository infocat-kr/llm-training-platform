"""
유틸리티 헬퍼 함수들
"""

import os
import random
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Union
import time
import json
import pickle
from pathlib import Path


def set_seed(seed: int = 42):
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    """초를 읽기 쉬운 시간 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}분 {secs:.1f}초"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}시간 {minutes}분 {secs:.1f}초"


def format_number(num: Union[int, float]) -> str:
    """숫자를 읽기 쉬운 형식으로 변환"""
    if isinstance(num, float):
        if num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.2f}"
    else:
        if num >= 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return str(num)


def create_directory(path: Union[str, Path], parents: bool = True) -> Path:
    """디렉토리 생성"""
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=True)
    return path


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """JSON 파일 저장"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Any:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """Pickle 파일 저장"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Pickle 파일 로드"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_file_size(filepath: Union[str, Path]) -> str:
    """파일 크기를 읽기 쉬운 형식으로 반환"""
    size = os.path.getsize(filepath)
    
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size/1024:.1f} KB"
    elif size < 1024**3:
        return f"{size/(1024**2):.1f} MB"
    else:
        return f"{size/(1024**3):.1f} GB"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """모델 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """모델 크기를 MB 단위로 계산"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def estimate_memory_usage(model: torch.nn.Module, batch_size: int, seq_length: int) -> Dict[str, float]:
    """메모리 사용량 추정 (MB)"""
    # 모델 파라미터 메모리
    model_memory = get_model_size_mb(model)
    
    # 활성화 메모리 (대략적 추정)
    # 각 레이어의 활성화 크기 = batch_size * seq_length * hidden_size
    hidden_size = getattr(model, 'd_model', 512)
    num_layers = getattr(model, 'num_layers', 6)
    
    activation_memory = batch_size * seq_length * hidden_size * num_layers * 4 / 1024**2  # float32
    
    # 그래디언트 메모리 (훈련 시)
    gradient_memory = model_memory
    
    # 옵티마이저 메모리 (Adam)
    optimizer_memory = model_memory * 2  # momentum + variance
    
    return {
        'model': model_memory,
        'activations': activation_memory,
        'gradients': gradient_memory,
        'optimizer': optimizer_memory,
        'total_training': model_memory + activation_memory + gradient_memory + optimizer_memory,
        'total_inference': model_memory + activation_memory
    }


def calculate_flops(model: torch.nn.Module, batch_size: int, seq_length: int) -> int:
    """FLOPs 계산 (대략적)"""
    # 어텐션 FLOPs
    hidden_size = getattr(model, 'd_model', 512)
    num_heads = getattr(model, 'num_heads', 8)
    num_layers = getattr(model, 'num_layers', 6)
    
    # Self-attention: 4 * batch_size * seq_length^2 * hidden_size
    attention_flops = 4 * batch_size * seq_length * seq_length * hidden_size * num_layers
    
    # Feed-forward: 2 * batch_size * seq_length * hidden_size * d_ff
    d_ff = getattr(model, 'd_ff', hidden_size * 4)
    ff_flops = 2 * batch_size * seq_length * hidden_size * d_ff * num_layers
    
    # Embedding: batch_size * seq_length * vocab_size * hidden_size
    vocab_size = getattr(model, 'vocab_size', 30000)
    embedding_flops = batch_size * seq_length * vocab_size * hidden_size
    
    total_flops = attention_flops + ff_flops + embedding_flops
    return total_flops


def benchmark_model(model: torch.nn.Module, input_shape: tuple, num_runs: int = 100) -> Dict[str, float]:
    """모델 성능 벤치마크"""
    model.eval()
    device = next(model.parameters()).device
    
    # 더미 입력 생성
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 벤치마크
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = input_shape[0] / avg_time  # samples per second
    
    return {
        'avg_inference_time': avg_time,
        'throughput': throughput,
        'fps': throughput
    }


def get_gpu_memory_info() -> Dict[str, float]:
    """GPU 메모리 정보 반환 (GB)"""
    if not torch.cuda.is_available():
        return {'available': False}
    
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        'available': True,
        'allocated': memory_allocated,
        'reserved': memory_reserved,
        'total': memory_total,
        'free': memory_total - memory_reserved
    }


def cleanup_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def validate_config(config: Dict[str, Any]) -> List[str]:
    """설정 유효성 검사"""
    errors = []
    
    # 필수 필드 검사
    required_fields = ['model_type', 'vocab_size', 'd_model', 'num_layers', 'num_heads']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # 값 범위 검사
    if 'd_model' in config and config['d_model'] <= 0:
        errors.append("d_model must be positive")
    
    if 'num_layers' in config and config['num_layers'] <= 0:
        errors.append("num_layers must be positive")
    
    if 'num_heads' in config and config['num_heads'] <= 0:
        errors.append("num_heads must be positive")
    
    if 'vocab_size' in config and config['vocab_size'] <= 0:
        errors.append("vocab_size must be positive")
    
    # 호환성 검사
    if 'd_model' in config and 'num_heads' in config:
        if config['d_model'] % config['num_heads'] != 0:
            errors.append("d_model must be divisible by num_heads")
    
    return errors


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """설정 병합 (깊은 병합)"""
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_config(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """중첩된 설정을 평면화"""
    items = []
    
    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def unflatten_config(flat_config: Dict[str, Any]) -> Dict[str, Any]:
    """평면화된 설정을 중첩 구조로 변환"""
    result = {}
    
    for key, value in flat_config.items():
        keys = key.split('.')
        current = result
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    return result
