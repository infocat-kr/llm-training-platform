"""
설정 로더
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, Union
from .config import Config


class ConfigLoader:
    """설정 로더 클래스"""
    
    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """JSON 파일 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    @staticmethod
    def save_json(config_dict: Dict[str, Any], filepath: str):
        """JSON 파일 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
    @staticmethod
    def load_yaml(filepath: str) -> Dict[str, Any]:
        """YAML 파일 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    @staticmethod
    def save_yaml(config_dict: Dict[str, Any], filepath: str):
        """YAML 파일 저장"""
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
    @staticmethod
    def load_config(filepath: str) -> Config:
        """설정 파일 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
            
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.json':
            config_dict = ConfigLoader.load_json(filepath)
        elif file_ext in ['.yaml', '.yml']:
            config_dict = ConfigLoader.load_yaml(filepath)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
            
        return Config.from_dict(config_dict)
        
    @staticmethod
    def save_config(config: Config, filepath: str):
        """설정 파일 저장"""
        config_dict = config.to_dict()
        
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.json':
            ConfigLoader.save_json(config_dict, filepath)
        elif file_ext in ['.yaml', '.yml']:
            ConfigLoader.save_yaml(config_dict, filepath)
        else:
            raise ValueError(f"Unsupported config file format: {file_ext}")
            
    @staticmethod
    def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
        """설정 병합"""
        merged_dict = base_config.to_dict()
        
        def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
                    
        deep_update(merged_dict, override_config)
        return Config.from_dict(merged_dict)


def load_config(filepath: str) -> Config:
    """설정 파일 로드 (편의 함수)"""
    return ConfigLoader.load_config(filepath)


def save_config(config: Config, filepath: str):
    """설정 파일 저장 (편의 함수)"""
    ConfigLoader.save_config(config, filepath)


def create_default_configs():
    """기본 설정 파일들 생성"""
    
    # 기본 설정
    config = Config()
    
    # 설정 디렉토리 생성
    configs_dir = './configs'
    os.makedirs(configs_dir, exist_ok=True)
    
    # 기본 설정 저장
    save_config(config, os.path.join(configs_dir, 'default.yaml'))
    
    # 모델별 설정 생성
    model_configs = {
        'gpt_small': {
            'model': {
                'model_type': 'gpt',
                'd_model': 256,
                'num_heads': 4,
                'num_layers': 4,
                'd_ff': 1024,
                'max_length': 512
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 1e-4,
                'epochs': 5
            }
        },
        'gpt_medium': {
            'model': {
                'model_type': 'gpt',
                'd_model': 512,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 2048,
                'max_length': 1024
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 5e-5,
                'epochs': 10
            }
        },
        'gpt_large': {
            'model': {
                'model_type': 'gpt',
                'd_model': 768,
                'num_heads': 12,
                'num_layers': 12,
                'd_ff': 3072,
                'max_length': 1024
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 3e-5,
                'epochs': 15
            }
        },
        'bert_base': {
            'model': {
                'model_type': 'bert',
                'd_model': 768,
                'num_heads': 12,
                'num_layers': 12,
                'd_ff': 3072,
                'max_length': 512
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'epochs': 3
            }
        },
        't5_small': {
            'model': {
                'model_type': 't5',
                'd_model': 512,
                'num_heads': 8,
                'num_layers': 6,
                'd_ff': 2048,
                'max_length': 512
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 1e-4,
                'epochs': 10
            }
        }
    }
    
    for model_name, model_config in model_configs.items():
        model_specific_config = ConfigLoader.merge_configs(config, model_config)
        save_config(model_specific_config, os.path.join(configs_dir, f'{model_name}.yaml'))
        
    # 작업별 설정 생성
    task_configs = {
        'language_modeling': {
            'data': {
                'dataset_type': 'language_model'
            },
            'training': {
                'loss_type': 'language_model'
            }
        },
        'text_classification': {
            'data': {
                'dataset_type': 'text'
            },
            'training': {
                'loss_type': 'classification'
            }
        },
        'conversation': {
            'data': {
                'dataset_type': 'conversation',
                'preprocessor_type': 'conversation'
            },
            'training': {
                'loss_type': 'language_model'
            }
        },
        'question_answering': {
            'data': {
                'dataset_type': 'qa'
            },
            'training': {
                'loss_type': 'language_model'
            }
        }
    }
    
    for task_name, task_config in task_configs.items():
        task_specific_config = ConfigLoader.merge_configs(config, task_config)
        save_config(task_specific_config, os.path.join(configs_dir, f'{task_name}.yaml'))
        
    print(f"Default configs created in {configs_dir}/")
    print("Available configs:")
    for filename in os.listdir(configs_dir):
        if filename.endswith('.yaml'):
            print(f"  - {filename}")


def load_config_with_overrides(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """오버라이드와 함께 설정 로드"""
    config = load_config(config_path)
    
    if overrides:
        config.update(overrides)
        
    return config


def validate_config_file(filepath: str) -> bool:
    """설정 파일 유효성 검사"""
    try:
        config = load_config(filepath)
        errors = config.validate()
        
        if errors:
            print(f"Config validation errors in {filepath}:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print(f"Config file {filepath} is valid")
            return True
            
    except Exception as e:
        print(f"Error loading config file {filepath}: {e}")
        return False


if __name__ == "__main__":
    # 기본 설정 파일들 생성
    create_default_configs()
