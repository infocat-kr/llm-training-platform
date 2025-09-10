"""
Flask 웹 애플리케이션
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import torch
import json
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TransformerModel
from data import SimpleTokenizer, TextDataset, create_dataloader
from training import Trainer, TrainingConfig
from inference import InferenceEngine, InferenceConfig, TextGenerator, GenerationConfig
from configs import load_config, Config


def create_app(config_path: Optional[str] = None) -> Flask:
    """Flask 애플리케이션 생성"""
    
    app = Flask(__name__)
    CORS(app)
    
    # 설정 로드
    if config_path and os.path.exists(config_path):
        app.config['LLM_CONFIG'] = load_config(config_path)
    else:
        app.config['LLM_CONFIG'] = Config()
        
    # 전역 변수들
    app.config['MODEL'] = None
    app.config['TOKENIZER'] = None
    app.config['INFERENCE_ENGINE'] = None
    
    @app.route('/')
    def index():
        """메인 페이지"""
        return render_template('index.html')
        
    @app.route('/train')
    def train_page():
        """훈련 페이지"""
        return render_template('train.html')
        
    @app.route('/inference')
    def inference_page():
        """추론 페이지"""
        return render_template('inference.html')
        
    @app.route('/chat')
    def chat_page():
        """채팅 페이지"""
        return render_template('chat.html')
        
    @app.route('/api/config', methods=['GET'])
    def get_config():
        """설정 조회"""
        config = app.config['LLM_CONFIG']
        return jsonify(config.to_dict())
        
    @app.route('/api/config', methods=['POST'])
    def update_config():
        """설정 업데이트"""
        try:
            updates = request.json
            config = app.config['LLM_CONFIG']
            config.update(updates)
            return jsonify({'status': 'success', 'message': 'Config updated'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
            
    @app.route('/api/model/info', methods=['GET'])
    def get_model_info():
        """모델 정보 조회"""
        model = app.config.get('MODEL')
        if model is None:
            return jsonify({'status': 'error', 'message': 'No model loaded'}), 400
            
        config = app.config['LLM_CONFIG']
        model_size = config.get_model_size()
        
        info = {
            'model_type': config.model.model_type,
            'total_parameters': model_size['total_params'],
            'd_model': config.model.d_model,
            'num_layers': config.model.num_layers,
            'num_heads': config.model.num_heads,
            'vocab_size': config.model.vocab_size
        }
        
        return jsonify(info)
        
    @app.route('/api/model/load', methods=['POST'])
    def load_model():
        """모델 로드"""
        try:
            data = request.json
            model_path = data.get('model_path')
            config_path = data.get('config_path')
            
            if config_path and os.path.exists(config_path):
                config = load_config(config_path)
                app.config['LLM_CONFIG'] = config
            else:
                config = app.config['LLM_CONFIG']
                
            # 모델 생성
            model = TransformerModel(
                model_type=config.model.model_type,
                vocab_size=config.model.vocab_size,
                d_model=config.model.d_model,
                num_heads=config.model.num_heads,
                num_layers=config.model.num_layers,
                d_ff=config.model.d_ff,
                max_len=config.model.max_length,
                dropout=config.model.dropout,
                activation=config.model.activation
            )
            
            # 체크포인트 로드
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                
            # 토크나이저 생성
            tokenizer = SimpleTokenizer(
                vocab_size=config.data.vocab_size,
                min_freq=config.data.min_freq
            )
            
            # 추론 엔진 생성
            inference_config = InferenceConfig(
                device=config.inference.device,
                batch_size=config.inference.batch_size,
                use_half_precision=config.inference.use_half_precision
            )
            
            inference_engine = InferenceEngine(model, tokenizer, inference_config)
            
            # 전역 변수에 저장
            app.config['MODEL'] = model
            app.config['TOKENIZER'] = tokenizer
            app.config['INFERENCE_ENGINE'] = inference_engine
            
            return jsonify({'status': 'success', 'message': 'Model loaded successfully'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
            
    @app.route('/api/train/start', methods=['POST'])
    def start_training():
        """훈련 시작"""
        try:
            data = request.json
            train_data_path = data.get('train_data_path')
            val_data_path = data.get('val_data_path')
            
            if not train_data_path or not os.path.exists(train_data_path):
                return jsonify({'status': 'error', 'message': 'Training data not found'}), 400
                
            config = app.config['LLM_CONFIG']
            
            # 토크나이저 생성
            tokenizer = SimpleTokenizer(
                vocab_size=config.data.vocab_size,
                min_freq=config.data.min_freq
            )
            
            # 데이터 로드 및 전처리
            with open(train_data_path, 'r', encoding='utf-8') as f:
                train_texts = f.readlines()
                
            # 토크나이저 훈련
            tokenizer.build_vocab(train_texts)
            
            # 데이터셋 생성
            train_dataset = TextDataset(
                texts=train_texts,
                tokenizer=tokenizer,
                max_length=config.data.max_length
            )
            
            val_dataset = None
            if val_data_path and os.path.exists(val_data_path):
                with open(val_data_path, 'r', encoding='utf-8') as f:
                    val_texts = f.readlines()
                val_dataset = TextDataset(
                    texts=val_texts,
                    tokenizer=tokenizer,
                    max_length=config.data.max_length
                )
                
            # 데이터로더 생성
            train_loader = create_dataloader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=True
            )
            
            val_loader = None
            if val_dataset:
                val_loader = create_dataloader(
                    val_dataset,
                    batch_size=config.training.batch_size,
                    shuffle=False
                )
                
            # 모델 생성
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
            
            # 훈련 설정
            training_config = TrainingConfig(
                epochs=config.training.epochs,
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                output_dir=config.training.output_dir,
                device=config.training.device,
                use_wandb=config.training.use_wandb,
                wandb_project=config.training.wandb_project
            )
            
            # 훈련 시작
            trainer = Trainer(
                model=model,
                config=training_config,
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            # 백그라운드에서 훈련 실행 (실제로는 Celery 등을 사용하는 것이 좋음)
            trainer.train()
            
            # 훈련된 모델 저장
            app.config['MODEL'] = model
            app.config['TOKENIZER'] = tokenizer
            
            return jsonify({'status': 'success', 'message': 'Training completed'})
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
            
    @app.route('/api/inference/generate', methods=['POST'])
    def generate_text():
        """텍스트 생성"""
        try:
            data = request.json
            input_text = data.get('input_text', '')
            max_length = data.get('max_length', 100)
            temperature = data.get('temperature', 1.0)
            top_p = data.get('top_p', 0.9)
            
            inference_engine = app.config.get('INFERENCE_ENGINE')
            if inference_engine is None:
                return jsonify({'status': 'error', 'message': 'No model loaded'}), 400
                
            # 생성 설정
            generation_config = GenerationConfig(
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            # 텍스트 생성
            result = inference_engine.predict(input_text, generation_config)
            
            return jsonify({
                'status': 'success',
                'generated_text': result['generated_texts'][0],
                'generation_time': result['generation_time']
            })
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
            
    @app.route('/api/chat/message', methods=['POST'])
    def chat_message():
        """채팅 메시지"""
        try:
            data = request.json
            message = data.get('message', '')
            user_id = data.get('user_id', 'default')
            
            inference_engine = app.config.get('INFERENCE_ENGINE')
            if inference_engine is None:
                return jsonify({'status': 'error', 'message': 'No model loaded'}), 400
                
            # 간단한 채팅 응답 생성
            prompt = f"Human: {message}\nAssistant:"
            result = inference_engine.predict(prompt)
            response = result['generated_texts'][0]
            
            return jsonify({
                'status': 'success',
                'response': response,
                'user_id': user_id
            })
            
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400
            
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """헬스 체크"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': app.config.get('MODEL') is not None,
            'tokenizer_loaded': app.config.get('TOKENIZER') is not None
        })
        
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
