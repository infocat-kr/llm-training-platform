#!/usr/bin/env python3
"""
LLM 모델 훈련 스크립트
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from models import TransformerModel
from data import SimpleTokenizer, TextDataset, create_dataloader, TextPreprocessor
from training import Trainer, TrainingConfig, create_optimizer, create_scheduler
from configs import load_config, Config


def main():
    parser = argparse.ArgumentParser(description='LLM 모델 훈련')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='설정 파일 경로')
    parser.add_argument('--train-data', type=str, required=True,
                       help='훈련 데이터 파일 경로')
    parser.add_argument('--val-data', type=str, default=None,
                       help='검증 데이터 파일 경로')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='출력 디렉토리')
    parser.add_argument('--model-type', type=str, default='gpt',
                       choices=['gpt', 'bert', 't5'],
                       help='모델 타입')
    parser.add_argument('--epochs', type=int, default=10,
                       help='훈련 에포크 수')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='학습률')
    parser.add_argument('--d-model', type=int, default=512,
                       help='모델 차원')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='레이어 수')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='어텐션 헤드 수')
    parser.add_argument('--vocab-size', type=int, default=30000,
                       help='어휘 크기')
    parser.add_argument('--max-length', type=int, default=512,
                       help='최대 시퀀스 길이')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto, cpu, cuda)')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Weights & Biases 사용')
    parser.add_argument('--wandb-project', type=str, default='llm-training',
                       help='W&B 프로젝트 이름')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Mixed Precision 훈련 사용')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='Gradient Checkpointing 사용')
    
    args = parser.parse_args()
    
    # 설정 로드
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = Config()
    
    # 명령행 인수로 설정 오버라이드
    config.model.model_type = args.model_type
    config.model.d_model = args.d_model
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.model.vocab_size = args.vocab_size
    config.model.max_length = args.max_length
    config.model.gradient_checkpointing = args.gradient_checkpointing
    
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.output_dir = args.output_dir
    config.training.device = args.device
    config.training.use_wandb = args.use_wandb
    config.training.wandb_project = args.wandb_project
    config.training.mixed_precision = args.mixed_precision
    
    config.data.vocab_size = args.vocab_size
    config.data.max_length = args.max_length
    
    print("=" * 60)
    print("LLM Training Platform")
    print("=" * 60)
    print(f"모델 타입: {config.model.model_type}")
    print(f"모델 크기: {config.model.d_model}D, {config.model.num_layers}L, {config.model.num_heads}H")
    print(f"어휘 크기: {config.model.vocab_size:,}")
    print(f"훈련 데이터: {args.train_data}")
    print(f"검증 데이터: {args.val_data}")
    print(f"에포크: {config.training.epochs}")
    print(f"배치 크기: {config.training.batch_size}")
    print(f"학습률: {config.training.learning_rate}")
    print(f"출력 디렉토리: {config.training.output_dir}")
    print("=" * 60)
    
    # 디바이스 설정
    if config.training.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.training.device)
    
    print(f"디바이스: {device}")
    
    # 데이터 로드
    print("\n데이터 로드 중...")
    with open(args.train_data, 'r', encoding='utf-8') as f:
        train_texts = f.readlines()
    
    print(f"훈련 데이터: {len(train_texts):,} 줄")
    
    val_texts = []
    if args.val_data and os.path.exists(args.val_data):
        with open(args.val_data, 'r', encoding='utf-8') as f:
            val_texts = f.readlines()
        print(f"검증 데이터: {len(val_texts):,} 줄")
    
    # 전처리
    print("\n텍스트 전처리 중...")
    preprocessor = TextPreprocessor(
        remove_punctuation=True,
        lowercase=True,
        min_length=1,
        max_length=config.data.max_length
    )
    
    train_texts = preprocessor.preprocess_batch(train_texts)
    if val_texts:
        val_texts = preprocessor.preprocess_batch(val_texts)
    
    # 토크나이저 생성 및 훈련
    print("\n토크나이저 생성 중...")
    tokenizer = SimpleTokenizer(
        vocab_size=config.data.vocab_size,
        min_freq=2
    )
    
    tokenizer.build_vocab(train_texts)
    print(f"어휘 크기: {len(tokenizer.vocab):,}")
    
    # 토크나이저 저장
    os.makedirs(config.training.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(config.training.output_dir, 'tokenizer.json'))
    
    # 데이터셋 생성
    print("\n데이터셋 생성 중...")
    train_dataset = TextDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=config.data.max_length
    )
    
    val_dataset = None
    if val_texts:
        val_dataset = TextDataset(
            texts=val_texts,
            tokenizer=tokenizer,
            max_length=config.data.max_length
        )
    
    # 데이터로더 생성
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory
        )
    
    # 모델 생성
    print("\n모델 생성 중...")
    model = TransformerModel(
        model_type=config.model.model_type,
        vocab_size=len(tokenizer.vocab),
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        d_ff=config.model.d_ff,
        max_len=config.model.max_length,
        dropout=config.model.dropout,
        activation=config.model.activation
    )
    
    # 모델 크기 정보
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터: {total_params:,}")
    print(f"훈련 가능한 파라미터: {trainable_params:,}")
    
    # 옵티마이저 및 스케줄러 생성
    optimizer = create_optimizer(
        model,
        optimizer_type=config.training.optimizer_type,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 훈련 스텝 수 계산
    num_training_steps = len(train_loader) * config.training.epochs
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_type=config.training.scheduler_type,
        num_training_steps=num_training_steps,
        num_warmup_steps=config.training.warmup_steps
    )
    
    # 훈련 설정
    training_config = TrainingConfig(
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        accumulation_steps=config.training.accumulation_steps,
        eval_steps=config.training.eval_steps,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        output_dir=config.training.output_dir,
        use_wandb=config.training.use_wandb,
        wandb_project=config.training.wandb_project,
        device=config.training.device,
        mixed_precision=config.training.mixed_precision,
        gradient_checkpointing=config.training.gradient_checkpointing
    )
    
    # 훈련 시작
    print("\n훈련 시작...")
    trainer = Trainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    try:
        trainer.train()
        print("\n훈련이 성공적으로 완료되었습니다!")
        
        # 최종 모델 저장
        final_model_path = os.path.join(config.training.output_dir, 'final_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'config': config,
            'total_params': total_params,
            'trainable_params': trainable_params
        }, final_model_path)
        
        print(f"모델이 저장되었습니다: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\n훈련이 중단되었습니다.")
        
        # 중단된 모델도 저장
        interrupted_model_path = os.path.join(config.training.output_dir, 'interrupted_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
            'config': config,
            'total_params': total_params,
            'trainable_params': trainable_params
        }, interrupted_model_path)
        
        print(f"중단된 모델이 저장되었습니다: {interrupted_model_path}")
        
    except Exception as e:
        print(f"\n훈련 중 오류가 발생했습니다: {e}")
        raise


if __name__ == "__main__":
    main()
