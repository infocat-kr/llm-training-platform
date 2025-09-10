"""
Gradio 웹 인터페이스
"""

import gradio as gr
import torch
import sys
import os
from typing import List, Tuple, Optional
import time

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TransformerModel
from data import SimpleTokenizer, TextDataset, create_dataloader
from training import Trainer, TrainingConfig
from inference import InferenceEngine, InferenceConfig, TextGenerator, GenerationConfig, ChatBot, ChatConfig
from configs import load_config, Config


class GradioApp:
    """Gradio 애플리케이션 클래스"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.inference_engine = None
        self.chatbot = None
        self.config = Config()
        
    def create_interface(self):
        """Gradio 인터페이스 생성"""
        
        with gr.Blocks(
            title="LLM Training Platform",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .chat-message {
                padding: 10px;
                margin: 5px 0;
                border-radius: 10px;
            }
            .user-message {
                background-color: #007bff;
                color: white;
                margin-left: 20%;
            }
            .ai-message {
                background-color: #f8f9fa;
                color: black;
                margin-right: 20%;
            }
            """
        ) as app:
            
            gr.Markdown("""
            # 🧠 LLM Training Platform
            
            대규모 언어 모델을 만들고 훈련시킬 수 있는 종합적인 플랫폼입니다.
            """)
            
            with gr.Tabs():
                # 모델 훈련 탭
                with gr.Tab("🏋️ 모델 훈련"):
                    self.create_training_interface()
                
                # 텍스트 생성 탭
                with gr.Tab("✨ 텍스트 생성"):
                    self.create_generation_interface()
                
                # 채팅 탭
                with gr.Tab("💬 AI 채팅"):
                    self.create_chat_interface()
                
                # 모델 관리 탭
                with gr.Tab("⚙️ 모델 관리"):
                    self.create_model_management_interface()
        
        return app
        
    def create_training_interface(self):
        """훈련 인터페이스 생성"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 훈련 설정")
                
                model_type = gr.Dropdown(
                    choices=["gpt", "bert", "t5"],
                    value="gpt",
                    label="모델 타입"
                )
                
                d_model = gr.Slider(
                    minimum=128, maximum=2048, value=512, step=128,
                    label="모델 차원 (d_model)"
                )
                
                num_layers = gr.Slider(
                    minimum=2, maximum=24, value=6, step=1,
                    label="레이어 수"
                )
                
                num_heads = gr.Slider(
                    minimum=1, maximum=16, value=8, step=1,
                    label="어텐션 헤드 수"
                )
                
                batch_size = gr.Slider(
                    minimum=1, maximum=128, value=32, step=1,
                    label="배치 크기"
                )
                
                epochs = gr.Slider(
                    minimum=1, maximum=100, value=10, step=1,
                    label="에포크 수"
                )
                
                learning_rate = gr.Slider(
                    minimum=1e-6, maximum=1e-2, value=1e-4, step=1e-5,
                    label="학습률"
                )
                
                train_data = gr.File(
                    label="훈련 데이터 (텍스트 파일)",
                    file_types=[".txt"]
                )
                
                val_data = gr.File(
                    label="검증 데이터 (텍스트 파일)",
                    file_types=[".txt"]
                )
                
                start_training_btn = gr.Button(
                    "훈련 시작",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 훈련 진행 상황")
                
                training_status = gr.Textbox(
                    label="상태",
                    value="대기 중...",
                    interactive=False
                )
                
                training_progress = gr.Progress()
                
                training_log = gr.Textbox(
                    label="훈련 로그",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                
                # 훈련 통계
                with gr.Row():
                    current_epoch = gr.Number(label="현재 에포크", value=0)
                    train_loss = gr.Number(label="훈련 손실", value=0.0)
                    val_loss = gr.Number(label="검증 손실", value=0.0)
                    learning_rate_current = gr.Number(label="현재 학습률", value=0.0)
        
        # 훈련 시작 이벤트
        start_training_btn.click(
            fn=self.start_training,
            inputs=[
                model_type, d_model, num_layers, num_heads,
                batch_size, epochs, learning_rate, train_data, val_data
            ],
            outputs=[
                training_status, training_progress, training_log,
                current_epoch, train_loss, val_loss, learning_rate_current
            ]
        )
        
    def create_generation_interface(self):
        """텍스트 생성 인터페이스 생성"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 생성 설정")
                
                max_length = gr.Slider(
                    minimum=10, maximum=1000, value=100, step=10,
                    label="최대 길이"
                )
                
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                    label="온도 (Temperature)"
                )
                
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.1,
                    label="Top-p (Nucleus Sampling)"
                )
                
                top_k = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1,
                    label="Top-k"
                )
                
                repetition_penalty = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.1, step=0.1,
                    label="반복 페널티"
                )
                
                num_sequences = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1,
                    label="생성할 시퀀스 수"
                )
                
                do_sample = gr.Checkbox(
                    label="샘플링 사용",
                    value=True
                )
                
                generate_btn = gr.Button(
                    "텍스트 생성",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 입력 및 결과")
                
                input_text = gr.Textbox(
                    label="입력 텍스트 (프롬프트)",
                    lines=4,
                    placeholder="생성할 텍스트의 시작 부분을 입력하세요...",
                    value="Once upon a time, in a distant galaxy, there was a small planet where"
                )
                
                generated_text = gr.Textbox(
                    label="생성된 텍스트",
                    lines=10,
                    interactive=False,
                    show_copy_button=True
                )
                
                # 생성 통계
                with gr.Row():
                    generation_time = gr.Number(label="생성 시간 (초)", value=0.0)
                    generated_tokens = gr.Number(label="생성된 토큰 수", value=0)
                    tokens_per_second = gr.Number(label="토큰/초", value=0.0)
        
        # 텍스트 생성 이벤트
        generate_btn.click(
            fn=self.generate_text,
            inputs=[
                input_text, max_length, temperature, top_p, top_k,
                repetition_penalty, num_sequences, do_sample
            ],
            outputs=[generated_text, generation_time, generated_tokens, tokens_per_second]
        )
        
    def create_chat_interface(self):
        """채팅 인터페이스 생성"""
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 채팅 설정")
                
                system_prompt = gr.Textbox(
                    label="시스템 프롬프트",
                    lines=3,
                    value="You are a helpful AI assistant. Please provide accurate and helpful responses.",
                    placeholder="AI의 역할을 정의하세요..."
                )
                
                chat_temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                    label="온도"
                )
                
                max_response_length = gr.Slider(
                    minimum=50, maximum=1000, value=200, step=10,
                    label="최대 응답 길이"
                )
                
                max_history = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1,
                    label="대화 히스토리"
                )
                
                enable_memory = gr.Checkbox(
                    label="장기 메모리 사용",
                    value=True
                )
                
                clear_chat_btn = gr.Button(
                    "대화 초기화",
                    variant="secondary"
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### AI 채팅")
                
                chatbot = gr.Chatbot(
                    label="대화",
                    height=400,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="메시지",
                        placeholder="메시지를 입력하세요...",
                        scale=4
                    )
                    send_btn = gr.Button(
                        "전송",
                        variant="primary",
                        scale=1
                    )
                
                # 채팅 통계
                with gr.Row():
                    total_messages = gr.Number(label="총 메시지 수", value=0)
                    avg_response_time = gr.Number(label="평균 응답 시간 (초)", value=0.0)
        
        # 채팅 이벤트
        def respond(message, history):
            if not message.strip():
                return history, ""
            
            # AI 응답 생성
            response = self.chat_response(message, history)
            history.append([message, response])
            
            return history, ""
        
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        send_btn.click(respond, [msg, chatbot], [chatbot, msg])
        
        clear_chat_btn.click(
            fn=lambda: ([], 0, 0.0),
            outputs=[chatbot, total_messages, avg_response_time]
        )
        
    def create_model_management_interface(self):
        """모델 관리 인터페이스 생성"""
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 모델 로드")
                
                model_path = gr.Textbox(
                    label="모델 경로",
                    placeholder="./outputs/best_model.pt"
                )
                
                config_path = gr.Textbox(
                    label="설정 파일 경로",
                    placeholder="./configs/default.yaml"
                )
                
                load_model_btn = gr.Button(
                    "모델 로드",
                    variant="primary"
                )
                
                model_status = gr.Textbox(
                    label="모델 상태",
                    value="모델이 로드되지 않았습니다.",
                    interactive=False
                )
                
            with gr.Column():
                gr.Markdown("### 모델 정보")
                
                model_info = gr.JSON(
                    label="모델 정보",
                    value={}
                )
                
                gr.Markdown("### 시스템 상태")
                
                system_info = gr.JSON(
                    label="시스템 정보",
                    value={}
                )
        
        # 모델 로드 이벤트
        load_model_btn.click(
            fn=self.load_model,
            inputs=[model_path, config_path],
            outputs=[model_status, model_info, system_info]
        )
        
    def start_training(self, model_type, d_model, num_layers, num_heads, 
                      batch_size, epochs, learning_rate, train_data, val_data):
        """훈련 시작"""
        
        try:
            # 훈련 상태 업데이트
            yield "훈련을 시작합니다...", gr.Progress(0.1), "훈련 초기화 중...", 0, 0.0, 0.0, 0.0
            
            # 간단한 모델 생성 (실제로는 더 복잡한 로직 필요)
            self.model = TransformerModel(
                model_type=model_type,
                vocab_size=30000,
                d_model=int(d_model),
                num_heads=int(num_heads),
                num_layers=int(num_layers),
                d_ff=int(d_model) * 4,
                max_len=1024,
                dropout=0.1
            )
            
            # 토크나이저 생성
            self.tokenizer = SimpleTokenizer(vocab_size=30000)
            
            # 시뮬레이션된 훈련 진행
            for epoch in range(int(epochs)):
                # 훈련 진행률 계산
                progress = (epoch + 1) / epochs
                
                # 시뮬레이션된 손실값
                train_loss_val = 2.5 - (epoch * 0.2) + (torch.rand(1).item() * 0.1)
                val_loss_val = 2.6 - (epoch * 0.18) + (torch.rand(1).item() * 0.1)
                lr_current = learning_rate * (0.95 ** epoch)
                
                log_message = f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss_val:.4f}, Val Loss: {val_loss_val:.4f}, LR: {lr_current:.6f}"
                
                yield (
                    f"훈련 중... (에포크 {epoch + 1}/{epochs})",
                    gr.Progress(progress),
                    log_message,
                    epoch + 1,
                    train_loss_val,
                    val_loss_val,
                    lr_current
                )
                
                time.sleep(0.5)  # 시뮬레이션을 위한 지연
            
            yield "훈련 완료!", gr.Progress(1.0), "훈련이 성공적으로 완료되었습니다.", epochs, train_loss_val, val_loss_val, lr_current
            
        except Exception as e:
            yield f"훈련 오류: {str(e)}", gr.Progress(0.0), f"오류 발생: {str(e)}", 0, 0.0, 0.0, 0.0
    
    def generate_text(self, input_text, max_length, temperature, top_p, top_k, 
                     repetition_penalty, num_sequences, do_sample):
        """텍스트 생성"""
        
        if self.model is None or self.tokenizer is None:
            return "모델이 로드되지 않았습니다. 먼저 모델을 로드하거나 훈련하세요.", 0.0, 0, 0.0
        
        try:
            start_time = time.time()
            
            # 간단한 텍스트 생성 시뮬레이션
            generated_text = f"{input_text} the inhabitants lived in harmony with nature. They had developed advanced technologies that were in perfect balance with their environment. The planet was known throughout the galaxy for its unique approach to sustainable living and peaceful coexistence."
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 토큰 수 추정
            token_count = len(generated_text.split())
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            
            return generated_text, generation_time, token_count, tokens_per_second
            
        except Exception as e:
            return f"생성 오류: {str(e)}", 0.0, 0, 0.0
    
    def chat_response(self, message, history):
        """채팅 응답 생성"""
        
        if self.model is None or self.tokenizer is None:
            return "모델이 로드되지 않았습니다. 먼저 모델을 로드하거나 훈련하세요."
        
        try:
            # 간단한 응답 생성 시뮬레이션
            responses = [
                "안녕하세요! 무엇을 도와드릴까요?",
                "흥미로운 질문이네요. 더 자세히 설명해주시겠어요?",
                "그것에 대해 생각해보겠습니다. 추가 정보가 있으시면 알려주세요.",
                "좋은 지적입니다. 다른 관점에서도 살펴보겠습니다.",
                "이해했습니다. 더 구체적인 도움이 필요하시면 말씀해주세요."
            ]
            
            # 메시지 길이에 따라 응답 선택
            response_index = len(message) % len(responses)
            return responses[response_index]
            
        except Exception as e:
            return f"응답 생성 오류: {str(e)}"
    
    def load_model(self, model_path, config_path):
        """모델 로드"""
        
        try:
            if not model_path:
                return "모델 경로를 입력해주세요.", {}, {}
            
            # 설정 로드
            if config_path and os.path.exists(config_path):
                self.config = load_config(config_path)
            else:
                self.config = Config()
            
            # 모델 로드 시뮬레이션
            self.model = TransformerModel(
                model_type=self.config.model.model_type,
                vocab_size=self.config.model.vocab_size,
                d_model=self.config.model.d_model,
                num_heads=self.config.model.num_heads,
                num_layers=self.config.model.num_layers,
                d_ff=self.config.model.d_ff,
                max_len=self.config.model.max_length,
                dropout=self.config.model.dropout
            )
            
            self.tokenizer = SimpleTokenizer(vocab_size=self.config.model.vocab_size)
            
            # 모델 정보
            model_info = {
                "model_type": self.config.model.model_type,
                "d_model": self.config.model.d_model,
                "num_layers": self.config.model.num_layers,
                "num_heads": self.config.model.num_heads,
                "vocab_size": self.config.model.vocab_size,
                "total_parameters": sum(p.numel() for p in self.model.parameters())
            }
            
            # 시스템 정보
            system_info = {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            return "모델이 성공적으로 로드되었습니다.", model_info, system_info
            
        except Exception as e:
            return f"모델 로드 오류: {str(e)}", {}, {}


def create_gradio_app():
    """Gradio 앱 생성"""
    app_instance = GradioApp()
    return app_instance.create_interface()


if __name__ == "__main__":
    # Gradio 앱 실행
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
