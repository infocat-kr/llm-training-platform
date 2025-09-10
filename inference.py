#!/usr/bin/env python3
"""
모델 추론 스크립트
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from models import TransformerModel
from data import SimpleTokenizer
from inference import InferenceEngine, InferenceConfig, TextGenerator, GenerationConfig


def main():
    parser = argparse.ArgumentParser(description='LLM 모델 추론')
    parser.add_argument('--model-path', type=str, required=True,
                       help='모델 파일 경로')
    parser.add_argument('--tokenizer-path', type=str, default=None,
                       help='토크나이저 파일 경로')
    parser.add_argument('--input-text', type=str, required=True,
                       help='입력 텍스트')
    parser.add_argument('--max-length', type=int, default=100,
                       help='최대 생성 길이')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='온도')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p 값')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k 값')
    parser.add_argument('--repetition-penalty', type=float, default=1.1,
                       help='반복 페널티')
    parser.add_argument('--num-sequences', type=int, default=1,
                       help='생성할 시퀀스 수')
    parser.add_argument('--do-sample', action='store_true',
                       help='샘플링 사용')
    parser.add_argument('--device', type=str, default='auto',
                       help='디바이스 (auto, cpu, cuda)')
    parser.add_argument('--output-file', type=str, default=None,
                       help='출력 파일 경로')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LLM Inference")
    print("=" * 60)
    print(f"모델 경로: {args.model_path}")
    print(f"입력 텍스트: {args.input_text}")
    print(f"최대 길이: {args.max_length}")
    print(f"온도: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print("=" * 60)
    
    # 디바이스 설정
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"디바이스: {device}")
    
    # 모델 로드
    print("\n모델 로드 중...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 모델 생성
    model_config = checkpoint.get('config', {})
    model = TransformerModel(
        model_type=model_config.get('model_type', 'gpt'),
        vocab_size=model_config.get('vocab_size', 30000),
        d_model=model_config.get('d_model', 512),
        num_heads=model_config.get('num_heads', 8),
        num_layers=model_config.get('num_layers', 6),
        d_ff=model_config.get('d_ff', 2048),
        max_len=model_config.get('max_length', 1024),
        dropout=model_config.get('dropout', 0.1),
        activation=model_config.get('activation', 'relu')
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("모델이 성공적으로 로드되었습니다.")
    
    # 토크나이저 로드
    print("\n토크나이저 로드 중...")
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        tokenizer = SimpleTokenizer()
        tokenizer.load(args.tokenizer_path)
    elif 'tokenizer' in checkpoint:
        tokenizer = checkpoint['tokenizer']
    else:
        # 기본 토크나이저 생성
        tokenizer = SimpleTokenizer(vocab_size=model_config.get('vocab_size', 30000))
        print("경고: 토크나이저를 찾을 수 없어 기본 토크나이저를 사용합니다.")
    
    print("토크나이저가 성공적으로 로드되었습니다.")
    
    # 추론 엔진 생성
    inference_config = InferenceConfig(
        device=device,
        batch_size=1,
        use_half_precision=False,
        enable_caching=True
    )
    
    inference_engine = InferenceEngine(model, tokenizer, inference_config)
    
    # 생성 설정
    generation_config = GenerationConfig(
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=args.num_sequences,
        do_sample=args.do_sample
    )
    
    # 텍스트 생성
    print("\n텍스트 생성 중...")
    try:
        result = inference_engine.predict(args.input_text, generation_config)
        
        print("\n생성 결과:")
        print("-" * 40)
        
        for i, generated_text in enumerate(result['generated_texts']):
            print(f"\n[시퀀스 {i+1}]")
            print(f"입력: {args.input_text}")
            print(f"생성: {generated_text}")
            print(f"전체: {args.input_text}{generated_text}")
        
        print(f"\n생성 시간: {result['generation_time']:.2f}초")
        
        # 출력 파일에 저장
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(f"입력: {args.input_text}\n\n")
                for i, generated_text in enumerate(result['generated_texts']):
                    f.write(f"생성 {i+1}: {generated_text}\n\n")
                f.write(f"생성 시간: {result['generation_time']:.2f}초\n")
            print(f"결과가 저장되었습니다: {args.output_file}")
        
    except Exception as e:
        print(f"생성 중 오류가 발생했습니다: {e}")
        raise


if __name__ == "__main__":
    main()
