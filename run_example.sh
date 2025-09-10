#!/bin/bash

# LLM Training Platform - 예제 실행 스크립트

echo "=========================================="
echo "LLM Training Platform - 예제 실행"
echo "=========================================="

# 가상환경 활성화 (선택사항)
# source venv/bin/activate

# 의존성 설치
echo "의존성 설치 중..."
pip install -r requirements.txt

# 샘플 데이터 확인
if [ ! -f "data/sample_data.txt" ]; then
    echo "샘플 데이터가 없습니다. data/sample_data.txt 파일을 확인하세요."
    exit 1
fi

echo "샘플 데이터 확인 완료"

# 기본 설정 파일 생성
echo "기본 설정 파일 생성 중..."
python -c "
from configs.config_loader import create_default_configs
create_default_configs()
print('기본 설정 파일이 생성되었습니다.')
"

# 출력 디렉토리 생성
mkdir -p outputs logs plots

echo ""
echo "=========================================="
echo "1. 모델 훈련 예제"
echo "=========================================="

# 모델 훈련 실행
python train.py \
    --train-data data/sample_data.txt \
    --model-type gpt \
    --d-model 256 \
    --num-layers 4 \
    --num-heads 4 \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --output-dir outputs/example_training

echo ""
echo "=========================================="
echo "2. 모델 추론 예제"
echo "=========================================="

# 모델 추론 실행
python inference.py \
    --model-path outputs/example_training/best_model.pt \
    --input-text "The future of artificial intelligence" \
    --max-length 50 \
    --temperature 0.8 \
    --output-file outputs/inference_result.txt

echo ""
echo "=========================================="
echo "3. 웹 인터페이스 실행"
echo "=========================================="

echo "Flask 웹 애플리케이션을 실행하려면:"
echo "python app.py"
echo ""
echo "Gradio 웹 애플리케이션을 실행하려면:"
echo "python gradio_app.py"
echo ""

echo "=========================================="
echo "실행 완료!"
echo "=========================================="
echo ""
echo "생성된 파일들:"
echo "- outputs/example_training/: 훈련된 모델"
echo "- outputs/inference_result.txt: 추론 결과"
echo "- logs/: 훈련 로그"
echo "- configs/: 설정 파일들"
echo ""
echo "웹 인터페이스 접속:"
echo "- Flask: http://localhost:5000"
echo "- Gradio: http://localhost:7860"
