#!/usr/bin/env python3
"""
Flask 웹 애플리케이션 실행 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from web.app import create_app

if __name__ == "__main__":
    # Flask 앱 생성
    app = create_app()
    
    # 개발 서버 실행
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
