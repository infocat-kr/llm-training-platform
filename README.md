# 🧠 LLM Training Platform

대규모 언어 모델(LLM)을 만들고 훈련시킬 수 있는 종합적인 플랫폼입니다.

## ✨ 주요 기능

- 🏗️ **다양한 모델 아키텍처**: GPT, BERT, T5 스타일 Transformer 모델 지원
- 📚 **데이터 처리**: 텍스트 전처리, 토크나이징, 데이터셋 관리
- 🚀 **고성능 훈련**: Mixed Precision, Gradient Accumulation, Learning Rate Scheduling
- 🎯 **추론 엔진**: 텍스트 생성, 채팅, 임베딩 추출
- 🌐 **웹 인터페이스**: Flask 및 Gradio 기반 사용자 친화적 UI
- 📊 **모니터링**: 실시간 훈련 모니터링, 메트릭 시각화
- ⚙️ **설정 관리**: YAML 기반 하이퍼파라미터 관리

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd llm_training_platform

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 예제 실행

```bash
# 예제 스크립트 실행
./run_example.sh

# 또는 개별 실행
python train.py --train-data data/sample_data.txt --epochs 5
python inference.py --model-path outputs/best_model.pt --input-text "Hello world"
```

### 3. 웹 인터페이스 실행

```bash
# Flask 웹 애플리케이션
python app.py
# 접속: http://localhost:5000

# Gradio 웹 애플리케이션
python gradio_app.py
# 접속: http://localhost:7860
```

## 📁 프로젝트 구조

```
llm_training_platform/
├── models/              # 모델 아키텍처
│   ├── transformer.py   # Transformer 모델들
│   ├── attention.py     # 어텐션 메커니즘
│   └── layers.py        # 기본 레이어들
├── data/                # 데이터 처리
│   ├── tokenizer.py     # 토크나이저
│   ├── dataset.py       # 데이터셋 클래스
│   └── preprocessing.py # 전처리
├── training/            # 훈련 로직
│   ├── trainer.py       # 훈련기
│   ├── optimizer.py     # 옵티마이저
│   └── loss.py          # 손실 함수
├── inference/           # 추론 엔진
│   ├── generator.py     # 텍스트 생성
│   ├── chat.py          # 채팅봇
│   └── evaluator.py     # 모델 평가
├── web/                 # 웹 인터페이스
│   ├── app.py           # Flask 앱
│   ├── gradio_app.py    # Gradio 앱
│   └── templates/       # HTML 템플릿
├── configs/             # 설정 파일
│   ├── config.py        # 설정 클래스
│   └── default.yaml     # 기본 설정
├── utils/               # 유틸리티
│   ├── helpers.py       # 헬퍼 함수
│   ├── logging.py       # 로깅
│   └── visualization.py # 시각화
├── notebooks/           # Jupyter 노트북
├── data/                # 데이터 파일
└── outputs/             # 출력 파일
```

## 🎯 사용 방법

### 명령행 인터페이스

#### 모델 훈련

```bash
python train.py \
    --train-data data/train.txt \
    --val-data data/val.txt \
    --model-type gpt \
    --d-model 512 \
    --num-layers 6 \
    --num-heads 8 \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --output-dir outputs/my_model
```

#### 모델 추론

```bash
python inference.py \
    --model-path outputs/my_model/best_model.pt \
    --input-text "Once upon a time" \
    --max-length 100 \
    --temperature 0.8 \
    --top-p 0.9
```

### Python API

```python
from models import TransformerModel
from data import SimpleTokenizer, TextDataset
from training import Trainer, TrainingConfig

# 모델 생성
model = TransformerModel(
    model_type='gpt',
    vocab_size=30000,
    d_model=512,
    num_heads=8,
    num_layers=6
)

# 토크나이저 생성
tokenizer = SimpleTokenizer(vocab_size=30000)
tokenizer.build_vocab(texts)

# 데이터셋 생성
dataset = TextDataset(texts, tokenizer, max_length=512)

# 훈련 실행
trainer = Trainer(model, TrainingConfig(), train_loader)
trainer.train()
```

### 웹 인터페이스

1. **Flask 웹 앱**: 완전한 기능의 웹 인터페이스
   - 모델 훈련 및 모니터링
   - 텍스트 생성 및 채팅
   - 설정 관리

2. **Gradio 앱**: 간단한 데모 인터페이스
   - 대화형 모델 훈련
   - 실시간 텍스트 생성
   - 채팅 인터페이스

## ⚙️ 설정

### 기본 설정 (configs/default.yaml)

```yaml
model:
  model_type: 'gpt'
  d_model: 512
  num_layers: 6
  num_heads: 8
  vocab_size: 30000

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  optimizer_type: 'adamw'
  scheduler_type: 'cosine'

inference:
  max_length: 100
  temperature: 1.0
  top_p: 0.9
```

### 모델별 프리셋

- `gpt_small.yaml`: 작은 GPT 모델 (256D, 4L)
- `gpt_medium.yaml`: 중간 GPT 모델 (512D, 6L)
- `gpt_large.yaml`: 큰 GPT 모델 (768D, 12L)
- `bert_base.yaml`: BERT 베이스 모델
- `t5_small.yaml`: T5 스몰 모델

## 📊 모니터링 및 시각화

### 훈련 모니터링

- 실시간 손실 및 메트릭 추적
- Weights & Biases 통합
- TensorBoard 지원
- 커스텀 로깅

### 시각화 도구

```python
from utils.visualization import plot_training_curves, plot_attention_weights

# 훈련 곡선 시각화
plot_training_curves('logs/training.json', save_plots=True)

# 어텐션 가중치 시각화
plot_attention_weights(attention_weights, tokens, layer=0, head=0)
```

## 🔧 고급 기능

### Mixed Precision 훈련

```bash
python train.py --mixed-precision
```

### Gradient Checkpointing

```bash
python train.py --gradient-checkpointing
```

### Weights & Biases 로깅

```bash
python train.py --use-wandb --wandb-project my-project
```

### 다중 GPU 훈련

```python
# DataParallel 사용
model = torch.nn.DataParallel(model)

# DistributedDataParallel 사용 (고급)
# torch.distributed.launch 또는 torchrun 사용
```

## 📈 성능 최적화

### 메모리 최적화

- Gradient Accumulation
- Gradient Checkpointing
- Mixed Precision Training
- 모델 병렬화

### 속도 최적화

- 컴파일된 모델 (torch.compile)
- 최적화된 데이터로더
- 효율적인 어텐션 구현

## 🧪 실험 및 평가

### 모델 평가

```python
from inference import ModelEvaluator

evaluator = ModelEvaluator(model, tokenizer)
metrics = evaluator.evaluate_language_model(test_data)
print(f"Perplexity: {metrics['perplexity']:.2f}")
```

### 벤치마킹

```python
from utils.helpers import benchmark_model

benchmark_results = benchmark_model(model, (batch_size, seq_len))
print(f"Throughput: {benchmark_results['throughput']:.2f} samples/sec")
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- PyTorch 팀
- Hugging Face Transformers
- OpenAI GPT 논문
- Google BERT 논문
- 모든 오픈소스 기여자들

## 📞 지원

- Issues: GitHub Issues 사용
- 문서: 프로젝트 Wiki 참조
- 토론: GitHub Discussions 사용

---

**LLM Training Platform**으로 당신만의 대규모 언어 모델을 만들어보세요! 🚀
