"""
시각화 유틸리티
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path


def plot_training_curves(
    log_file: str,
    output_dir: str = './plots',
    save_plots: bool = True,
    show_plots: bool = False
) -> Dict[str, str]:
    """훈련 곡선 시각화"""
    
    # 로그 파일 로드
    with open(log_file, 'r', encoding='utf-8') as f:
        training_log = json.load(f)
    
    if not training_log:
        print("훈련 로그가 비어있습니다.")
        return {}
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 데이터 추출
    steps = [entry['step'] for entry in training_log]
    losses = [entry['loss'] for entry in training_log]
    learning_rates = [entry.get('learning_rate', 0) for entry in training_log]
    
    # 플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # 손실 곡선
    axes[0, 0].plot(steps, losses, 'b-', alpha=0.7)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 학습률 곡선
    axes[0, 1].plot(steps, learning_rates, 'r-', alpha=0.7)
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 손실 분포
    axes[1, 0].hist(losses, bins=50, alpha=0.7, color='green')
    axes[1, 0].set_title('Loss Distribution')
    axes[1, 0].set_xlabel('Loss')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 이동 평균 손실
    window_size = min(100, len(losses) // 10)
    if window_size > 1:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        axes[1, 1].plot(steps[window_size-1:], moving_avg, 'purple', linewidth=2)
        axes[1, 1].set_title(f'Moving Average Loss (window={window_size})')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    saved_files = {}
    if save_plots:
        plot_file = output_path / 'training_curves.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        saved_files['training_curves'] = str(plot_file)
    
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return saved_files


def plot_attention_weights(
    attention_weights: torch.Tensor,
    tokens: List[str],
    layer: int = 0,
    head: int = 0,
    output_dir: str = './plots',
    save_plot: bool = True,
    show_plot: bool = False
) -> Optional[str]:
    """어텐션 가중치 시각화"""
    
    # 어텐션 가중치 추출
    if attention_weights.dim() == 4:  # (batch, heads, seq, seq)
        attn = attention_weights[0, head, :, :].detach().cpu().numpy()
    elif attention_weights.dim() == 3:  # (heads, seq, seq)
        attn = attention_weights[head, :, :].detach().cpu().numpy()
    else:  # (seq, seq)
        attn = attention_weights.detach().cpu().numpy()
    
    # 플롯 생성
    plt.figure(figsize=(12, 10))
    
    # 히트맵
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        cbar=True,
        square=True
    )
    
    plt.title(f'Attention Weights - Layer {layer}, Head {head}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # 저장
    saved_file = None
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_file = output_path / f'attention_layer{layer}_head{head}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        saved_file = str(plot_file)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_file


def plot_model_comparison(
    model_results: Dict[str, Dict[str, List[float]]],
    output_dir: str = './plots',
    save_plot: bool = True,
    show_plot: bool = False
) -> Optional[str]:
    """모델 비교 시각화"""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (model_name, results) in enumerate(model_results.items()):
        if 'loss' in results:
            epochs = range(1, len(results['loss']) + 1)
            plt.plot(epochs, results['loss'], 
                    color=colors[i % len(colors)], 
                    label=f'{model_name} (Loss)',
                    linewidth=2)
    
    plt.title('Model Comparison - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    saved_file = None
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_file = output_path / 'model_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        saved_file = str(plot_file)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_file


def plot_learning_rate_schedule(
    optimizer,
    total_steps: int,
    output_dir: str = './plots',
    save_plot: bool = True,
    show_plot: bool = False
) -> Optional[str]:
    """학습률 스케줄 시각화"""
    
    # 학습률 히스토리 수집
    lr_history = []
    steps = []
    
    # 옵티마이저 상태 백업
    original_state = optimizer.state_dict()
    
    # 초기 학습률
    initial_lr = optimizer.param_groups[0]['lr']
    
    # 스케줄 시뮬레이션
    for step in range(0, total_steps, max(1, total_steps // 1000)):
        # 옵티마이저 초기화
        optimizer.load_state_dict(original_state)
        
        # 스케줄러 적용
        for _ in range(step):
            if hasattr(optimizer, 'scheduler'):
                optimizer.scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        steps.append(step)
    
    # 원래 상태 복원
    optimizer.load_state_dict(original_state)
    
    # 플롯 생성
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lr_history, 'b-', linewidth=2)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    # 저장
    saved_file = None
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_file = output_path / 'learning_rate_schedule.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        saved_file = str(plot_file)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_file


def plot_token_distribution(
    tokens: List[str],
    counts: List[int],
    top_k: int = 50,
    output_dir: str = './plots',
    save_plot: bool = True,
    show_plot: bool = False
) -> Optional[str]:
    """토큰 분포 시각화"""
    
    # 상위 k개 토큰 선택
    token_counts = list(zip(tokens, counts))
    token_counts.sort(key=lambda x: x[1], reverse=True)
    top_tokens, top_counts = zip(*token_counts[:top_k])
    
    # 플롯 생성
    plt.figure(figsize=(15, 8))
    
    bars = plt.bar(range(len(top_tokens)), top_counts, color='skyblue', alpha=0.7)
    plt.title(f'Top {top_k} Token Distribution')
    plt.xlabel('Tokens')
    plt.ylabel('Count')
    plt.xticks(range(len(top_tokens)), top_tokens, rotation=45, ha='right')
    
    # 값 표시
    for i, (bar, count) in enumerate(zip(bars, top_counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(top_counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 저장
    saved_file = None
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_file = output_path / 'token_distribution.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        saved_file = str(plot_file)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_file


def plot_embedding_visualization(
    embeddings: torch.Tensor,
    labels: Optional[List[str]] = None,
    method: str = 'tsne',
    output_dir: str = './plots',
    save_plot: bool = True,
    show_plot: bool = False
) -> Optional[str]:
    """임베딩 시각화 (t-SNE 또는 PCA)"""
    
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # 임베딩을 numpy로 변환
    if embeddings.dim() > 2:
        embeddings = embeddings.view(embeddings.size(0), -1)
    
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # 차원 축소
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    embeddings_2d = reducer.fit_transform(embeddings_np)
    
    # 플롯 생성
    plt.figure(figsize=(10, 8))
    
    if labels:
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7)
        
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(f'Embedding Visualization ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    saved_file = None
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_file = output_path / f'embedding_{method}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        saved_file = str(plot_file)
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return saved_file


def create_training_report(
    log_files: List[str],
    output_dir: str = './reports',
    save_report: bool = True
) -> Optional[str]:
    """훈련 리포트 생성"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 로그 파일 처리
    all_plots = {}
    
    for log_file in log_files:
        log_name = Path(log_file).stem
        plots = plot_training_curves(log_file, str(output_path), save_plots=True, show_plots=False)
        all_plots[log_name] = plots
    
    # HTML 리포트 생성
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #666; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
            .plot-container { margin: 30px 0; }
        </style>
    </head>
    <body>
        <h1>LLM Training Report</h1>
        <p>Generated on: {timestamp}</p>
    """
    
    for log_name, plots in all_plots.items():
        html_content += f"<h2>{log_name}</h2>"
        for plot_name, plot_path in plots.items():
            html_content += f"""
            <div class="plot-container">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{Path(plot_path).name}" alt="{plot_name}">
            </div>
            """
    
    html_content += """
    </body>
    </html>
    """
    
    # 리포트 저장
    if save_report:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f'training_report_{timestamp}.html'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content.format(timestamp=timestamp))
        
        return str(report_file)
    
    return None
