#!/usr/bin/env python3
"""
Gradio 웹 애플리케이션 실행 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from web.gradio_app import create_gradio_app

if __name__ == "__main__":
    # Gradio 앱 생성
    app = create_gradio_app()
    
    # Gradio 서버 실행
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )
