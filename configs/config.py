"""
설정 클래스들
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import os


@dataclass
class ModelConfig:
    """모델 설정"""
    # 모델 타입
    model_type: str = 'gpt'  # 'gpt', 'bert', 't5'
    
    # 모델 크기
    vocab_size: int = 30000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    max_length: int = 1024
    
    # 드롭아웃 및 정규화
    dropout: float = 0.1
    activation: str = 'relu'  # 'relu', 'gelu', 'swish'
    
    # 특수 토큰
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1
    unk_token_id: int = 3
    
    # 추가 설정
    use_bias: bool = True
    layer_norm_eps: float = 1e-6
    gradient_checkpointing: bool = False


@dataclass
class DataConfig:
    """데이터 설정"""
    # 데이터 경로
    train_data_path: str = './data/train.txt'
    val_data_path: str = './data/val.txt'
    test_data_path: str = './data/test.txt'
    
    # 데이터셋 타입
    dataset_type: str = 'text'  # 'text', 'language_model', 'conversation', 'qa'
    
    # 토크나이저 설정
    tokenizer_type: str = 'simple'  # 'simple', 'bpe'
    tokenizer_path: Optional[str] = None
    vocab_size: int = 30000
    min_freq: int = 2
    
    # 전처리 설정
    preprocessor_type: str = 'default'  # 'default', 'conversation', 'code'
    remove_punctuation: bool = True
    remove_numbers: bool = False
    remove_stopwords: bool = False
    lowercase: bool = True
    min_length: int = 1
    max_length: int = 512
    
    # 데이터 분할
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """훈련 설정"""
    # 기본 훈련 설정
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # 옵티마이저 설정
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd', 'rmsprop'
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # 스케줄러 설정
    scheduler_type: str = 'cosine'  # 'cosine', 'linear', 'step', 'exponential'
    warmup_steps: int = 1000
    num_training_steps: Optional[int] = None
    
    # 정규화 및 안정성
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = False
    
    # 손실 함수
    loss_type: str = 'language_model'  # 'language_model', 'classification', 'cross_entropy'
    label_smoothing: float = 0.0
    ignore_index: int = -100
    
    # 평가 및 저장
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    output_dir: str = './outputs'
    
    # 모니터링
    use_wandb: bool = False
    wandb_project: str = 'llm-training'
    use_tensorboard: bool = True
    
    # 디바이스
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class InferenceConfig:
    """추론 설정"""
    # 생성 설정
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
    
    # 추론 엔진 설정
    device: str = 'auto'
    batch_size: int = 1
    max_memory_usage: float = 0.8
    use_half_precision: bool = False
    enable_caching: bool = True
    cache_size: int = 1000
    
    # 채팅 설정
    system_prompt: str = "You are a helpful AI assistant."
    max_history: int = 10
    enable_memory: bool = True
    memory_size: int = 100


@dataclass
class Config:
    """전체 설정"""
    # 하위 설정들
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # 메타데이터
    name: str = 'llm_training'
    version: str = '1.0.0'
    description: str = 'LLM Training Configuration'
    author: str = 'LLM Training Platform'
    
    # 경로 설정
    project_root: str = '.'
    data_dir: str = './data'
    output_dir: str = './outputs'
    log_dir: str = './logs'
    
    def __post_init__(self):
        """초기화 후 처리"""
        # 경로 정규화
        self.project_root = os.path.abspath(self.project_root)
        self.data_dir = os.path.abspath(self.data_dir)
        self.output_dir = os.path.abspath(self.output_dir)
        self.log_dir = os.path.abspath(self.log_dir)
        
        # 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 하위 설정들의 경로 업데이트
        self.training.output_dir = os.path.join(self.output_dir, 'training')
        self.data.train_data_path = os.path.join(self.data_dir, 'train.txt')
        self.data.val_data_path = os.path.join(self.data_dir, 'val.txt')
        self.data.test_data_path = os.path.join(self.data_dir, 'test.txt')
        
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__,
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'project_root': self.project_root,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """딕셔너리에서 생성"""
        # 하위 설정들 생성
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        inference_config = InferenceConfig(**config_dict.get('inference', {}))
        
        # 메인 설정 생성
        config = cls(
            model=model_config,
            data=data_config,
            training=training_config,
            inference=inference_config,
            name=config_dict.get('name', 'llm_training'),
            version=config_dict.get('version', '1.0.0'),
            description=config_dict.get('description', 'LLM Training Configuration'),
            author=config_dict.get('author', 'LLM Training Platform'),
            project_root=config_dict.get('project_root', '.'),
            data_dir=config_dict.get('data_dir', './data'),
            output_dir=config_dict.get('output_dir', './outputs'),
            log_dir=config_dict.get('log_dir', './logs')
        )
        
        return config
        
    def update(self, updates: Dict[str, Any]):
        """설정 업데이트"""
        for key, value in updates.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(getattr(self, key), '__dict__'):
                    # 하위 설정 업데이트
                    sub_config = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(sub_config, sub_key):
                            setattr(sub_config, sub_key, sub_value)
                else:
                    # 직접 속성 업데이트
                    setattr(self, key, value)
                    
    def validate(self) -> List[str]:
        """설정 유효성 검사"""
        errors = []
        
        # 모델 설정 검사
        if self.model.d_model % self.model.num_heads != 0:
            errors.append("d_model must be divisible by num_heads")
            
        if self.model.vocab_size <= 0:
            errors.append("vocab_size must be positive")
            
        # 데이터 설정 검사
        if not 0 < self.data.train_ratio < 1:
            errors.append("train_ratio must be between 0 and 1")
            
        if not 0 < self.data.val_ratio < 1:
            errors.append("val_ratio must be between 0 and 1")
            
        if abs(self.data.train_ratio + self.data.val_ratio + self.data.test_ratio - 1.0) > 1e-6:
            errors.append("train_ratio + val_ratio + test_ratio must equal 1.0")
            
        # 훈련 설정 검사
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
            
        if self.training.batch_size <= 0:
            errors.append("batch_size must be positive")
            
        if self.training.epochs <= 0:
            errors.append("epochs must be positive")
            
        # 추론 설정 검사
        if self.inference.temperature <= 0:
            errors.append("temperature must be positive")
            
        if self.inference.max_length <= 0:
            errors.append("max_length must be positive")
            
        return errors
        
    def get_model_size(self) -> Dict[str, int]:
        """모델 크기 정보 반환"""
        # 대략적인 파라미터 수 계산
        embedding_params = self.model.vocab_size * self.model.d_model
        attention_params = 4 * self.model.d_model * self.model.d_model * self.model.num_layers
        ff_params = 2 * self.model.d_model * self.model.d_ff * self.model.num_layers
        output_params = self.model.d_model * self.model.vocab_size
        
        total_params = embedding_params + attention_params + ff_params + output_params
        
        return {
            'embedding_params': embedding_params,
            'attention_params': attention_params,
            'ff_params': ff_params,
            'output_params': output_params,
            'total_params': total_params
        }
        
    def get_training_steps(self, dataset_size: int) -> int:
        """훈련 스텝 수 계산"""
        steps_per_epoch = dataset_size // (self.training.batch_size * self.training.accumulation_steps)
        total_steps = steps_per_epoch * self.training.epochs
        return total_steps
