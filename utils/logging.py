"""
로깅 유틸리티
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
from datetime import datetime


def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    include_timestamp: bool = True
) -> logging.Logger:
    """로깅 설정"""
    
    # 로그 레벨 설정
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 로그 포맷 설정
    if log_format is None:
        if include_timestamp:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            log_format = '%(name)s - %(levelname)s - %(message)s'
    
    # 로거 생성
    logger = logging.getLogger('llm_training')
    logger.setLevel(numeric_level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 생성
    formatter = logging.Formatter(log_format)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'llm_training') -> logging.Logger:
    """로거 가져오기"""
    return logging.getLogger(name)


class TrainingLogger:
    """훈련 로깅 클래스"""
    
    def __init__(self, log_dir: str = './logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_log = []
        self.metrics_log = []
        
        # 로그 파일 경로
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.training_log_file = self.log_dir / f'training_{timestamp}.json'
        self.metrics_log_file = self.log_dir / f'metrics_{timestamp}.json'
        
    def log_training_step(self, step: int, epoch: int, loss: float, 
                         learning_rate: float, **kwargs):
        """훈련 스텝 로깅"""
        log_entry = {
            'step': step,
            'epoch': epoch,
            'loss': loss,
            'learning_rate': learning_rate,
            'timestamp': time.time(),
            **kwargs
        }
        
        self.training_log.append(log_entry)
        
        # 주기적으로 파일에 저장
        if step % 100 == 0:
            self.save_training_log()
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                   val_metrics: Optional[Dict[str, float]] = None):
        """메트릭 로깅"""
        log_entry = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'timestamp': time.time()
        }
        
        self.metrics_log.append(log_entry)
        self.save_metrics_log()
    
    def save_training_log(self):
        """훈련 로그 저장"""
        with open(self.training_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_log, f, indent=2, ensure_ascii=False)
    
    def save_metrics_log(self):
        """메트릭 로그 저장"""
        with open(self.metrics_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_log, f, indent=2, ensure_ascii=False)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """훈련 요약 반환"""
        if not self.training_log:
            return {}
        
        losses = [entry['loss'] for entry in self.training_log]
        
        return {
            'total_steps': len(self.training_log),
            'final_loss': losses[-1],
            'min_loss': min(losses),
            'max_loss': max(losses),
            'avg_loss': sum(losses) / len(losses),
            'training_time': self.training_log[-1]['timestamp'] - self.training_log[0]['timestamp'] if len(self.training_log) > 1 else 0
        }


class MetricsTracker:
    """메트릭 추적 클래스"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """메트릭 업데이트"""
        if step is None:
            step = len(self.history)
        
        entry = {
            'step': step,
            'timestamp': time.time(),
            'metrics': metrics.copy()
        }
        
        self.history.append(entry)
        
        # 현재 메트릭 업데이트
        self.metrics.update(metrics)
    
    def get_metric(self, name: str) -> Optional[float]:
        """특정 메트릭 값 반환"""
        return self.metrics.get(name)
    
    def get_metric_history(self, name: str) -> list:
        """특정 메트릭의 히스토리 반환"""
        return [entry['metrics'].get(name) for entry in self.history 
                if name in entry['metrics']]
    
    def get_best_metric(self, name: str, mode: str = 'min') -> tuple:
        """최고 메트릭 값과 스텝 반환"""
        history = self.get_metric_history(name)
        if not history:
            return None, None
        
        if mode == 'min':
            best_value = min(history)
            best_idx = history.index(best_value)
        else:
            best_value = max(history)
            best_idx = history.index(best_value)
        
        best_step = self.history[best_idx]['step']
        return best_value, best_step
    
    def reset(self):
        """메트릭 초기화"""
        self.metrics = {}
        self.history = []


class ProgressLogger:
    """진행 상황 로깅 클래스"""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.last_update_time = self.start_time
    
    def update(self, n: int = 1, description: Optional[str] = None):
        """진행 상황 업데이트"""
        self.current += n
        
        if description:
            self.description = description
        
        # 1초마다 또는 완료 시 업데이트
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0 or self.current >= self.total:
            self._print_progress()
            self.last_update_time = current_time
    
    def _print_progress(self):
        """진행 상황 출력"""
        if self.total == 0:
            return
        
        progress = self.current / self.total
        bar_length = 50
        filled_length = int(bar_length * progress)
        
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed_time * (self.total - self.current) / self.current
            eta_str = f"ETA: {self._format_time(eta)}"
        else:
            eta_str = "ETA: --:--"
        
        print(f"\r{self.description}: |{bar}| {progress:.1%} "
              f"({self.current}/{self.total}) "
              f"Elapsed: {self._format_time(elapsed_time)} {eta_str}", 
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # 새 줄
    
    def _format_time(self, seconds: float) -> str:
        """시간 포맷팅"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"


def log_model_info(model, logger: Optional[logging.Logger] = None):
    """모델 정보 로깅"""
    if logger is None:
        logger = get_logger()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Non-trainable: {total_params - trainable_params:,}")
    
    # 모델 크기
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024**2
    
    logger.info(f"Model Size: {model_size_mb:.2f} MB")


def log_training_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """훈련 설정 로깅"""
    if logger is None:
        logger = get_logger()
    
    logger.info("Training Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")


def log_system_info(logger: Optional[logging.Logger] = None):
    """시스템 정보 로깅"""
    if logger is None:
        logger = get_logger()
    
    import torch
    import platform
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
