"""
채팅봇 구현
"""

import torch
from typing import List, Dict, Any, Optional, Callable
import json
import time
from dataclasses import dataclass
from .generator import TextGenerator, GenerationConfig


@dataclass
class ChatConfig:
    """채팅 설정"""
    system_prompt: str = "You are a helpful AI assistant."
    max_history: int = 10
    temperature: float = 0.7
    max_length: int = 200
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    enable_memory: bool = True
    memory_size: int = 100


class ChatBot:
    """채팅봇"""
    
    def __init__(self, model, tokenizer, config: ChatConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ChatConfig()
        
        # 텍스트 생성기 초기화
        generation_config = GenerationConfig(
            temperature=self.config.temperature,
            max_length=self.config.max_length,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repetition_penalty=self.config.repetition_penalty
        )
        self.generator = TextGenerator(model, tokenizer, generation_config)
        
        # 대화 히스토리
        self.history = []
        
        # 메모리 (장기 기억)
        if self.config.enable_memory:
            self.memory = []
            
    def chat(self, 
             message: str,
             user_id: Optional[str] = None,
             session_id: Optional[str] = None) -> str:
        """채팅 응답"""
        
        # 시스템 프롬프트 추가
        if not self.history:
            self.history.append({
                'role': 'system',
                'content': self.config.system_prompt
            })
            
        # 사용자 메시지 추가
        self.history.append({
            'role': 'user',
            'content': message,
            'timestamp': time.time(),
            'user_id': user_id,
            'session_id': session_id
        })
        
        # 프롬프트 생성
        prompt = self._format_prompt()
        
        # 응답 생성
        response = self.generator.generate(prompt)[0]
        
        # 응답 정리
        response = self._clean_response(response)
        
        # 어시스턴트 응답 추가
        self.history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': time.time(),
            'user_id': user_id,
            'session_id': session_id
        })
        
        # 히스토리 길이 제한
        self._trim_history()
        
        # 메모리에 중요한 정보 저장
        if self.config.enable_memory:
            self._update_memory(message, response)
            
        return response
        
    def chat_stream(self, 
                   message: str,
                   callback: Callable[[str], None],
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None):
        """스트리밍 채팅"""
        
        # 사용자 메시지 추가
        self.history.append({
            'role': 'user',
            'content': message,
            'timestamp': time.time(),
            'user_id': user_id,
            'session_id': session_id
        })
        
        # 프롬프트 생성
        prompt = self._format_prompt()
        
        # 스트리밍 생성
        full_response = ""
        
        def stream_callback(token, full_text):
            nonlocal full_response
            full_response = full_text
            callback(token)
            
        self.generator.generate_stream(prompt, callback=stream_callback)
        
        # 응답 정리
        response = self._clean_response(full_response)
        
        # 어시스턴트 응답 추가
        self.history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': time.time(),
            'user_id': user_id,
            'session_id': session_id
        })
        
        # 히스토리 길이 제한
        self._trim_history()
        
        # 메모리 업데이트
        if self.config.enable_memory:
            self._update_memory(message, response)
            
    def _format_prompt(self) -> str:
        """대화 히스토리를 프롬프트로 포맷팅"""
        prompt = ""
        
        for turn in self.history:
            if turn['role'] == 'system':
                prompt += f"System: {turn['content']}\n\n"
            elif turn['role'] == 'user':
                prompt += f"Human: {turn['content']}\n"
            elif turn['role'] == 'assistant':
                prompt += f"Assistant: {turn['content']}\n\n"
                
        prompt += "Assistant:"
        
        return prompt
        
    def _clean_response(self, response: str) -> str:
        """응답 정리"""
        # 불필요한 접두사 제거
        prefixes_to_remove = [
            "Assistant:", "AI:", "Bot:", "Response:",
            "Here's", "Here is", "I'll", "I will"
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                
        # 불필요한 접미사 제거
        suffixes_to_remove = [
            "\n\nHuman:", "\nHuman:", "Human:",
            "\n\nSystem:", "\nSystem:", "System:"
        ]
        
        for suffix in suffixes_to_remove:
            if suffix in response:
                response = response.split(suffix)[0].strip()
                
        return response.strip()
        
    def _trim_history(self):
        """히스토리 길이 제한"""
        if len(self.history) > self.config.max_history * 2 + 1:  # system + user/assistant pairs
            # 시스템 메시지와 최근 대화만 유지
            system_msg = self.history[0] if self.history[0]['role'] == 'system' else None
            recent_history = self.history[-(self.config.max_history * 2):]
            
            self.history = []
            if system_msg:
                self.history.append(system_msg)
            self.history.extend(recent_history)
            
    def _update_memory(self, user_message: str, assistant_response: str):
        """메모리 업데이트"""
        # 중요한 정보 추출 (간단한 구현)
        important_info = self._extract_important_info(user_message, assistant_response)
        
        if important_info:
            self.memory.append({
                'info': important_info,
                'timestamp': time.time(),
                'context': {
                    'user_message': user_message,
                    'assistant_response': assistant_response
                }
            })
            
        # 메모리 크기 제한
        if len(self.memory) > self.config.memory_size:
            self.memory = self.memory[-self.config.memory_size:]
            
    def _extract_important_info(self, user_message: str, assistant_response: str) -> Optional[str]:
        """중요한 정보 추출 (간단한 구현)"""
        # 이름, 날짜, 숫자 등 중요한 정보 추출
        import re
        
        # 이름 패턴 (대문자로 시작하는 단어)
        names = re.findall(r'\b[A-Z][a-z]+\b', user_message)
        
        # 날짜 패턴
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', user_message)
        
        # 전화번호 패턴
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', user_message)
        
        # 이메일 패턴
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
        
        important_info = []
        if names:
            important_info.append(f"Names mentioned: {', '.join(names)}")
        if dates:
            important_info.append(f"Dates mentioned: {', '.join(dates)}")
        if phones:
            important_info.append(f"Phone numbers: {', '.join(phones)}")
        if emails:
            important_info.append(f"Email addresses: {', '.join(emails)}")
            
        return '; '.join(important_info) if important_info else None
        
    def get_memory_context(self, limit: int = 5) -> str:
        """메모리 컨텍스트 반환"""
        if not self.memory:
            return ""
            
        recent_memory = self.memory[-limit:]
        context = "Previous conversation context:\n"
        
        for mem in recent_memory:
            context += f"- {mem['info']}\n"
            
        return context
        
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.history = []
        
    def clear_memory(self):
        """메모리 초기화"""
        if hasattr(self, 'memory'):
            self.memory = []
            
    def save_conversation(self, filepath: str):
        """대화 저장"""
        conversation_data = {
            'history': self.history,
            'memory': getattr(self, 'memory', []),
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
    def load_conversation(self, filepath: str):
        """대화 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
            
        self.history = conversation_data.get('history', [])
        self.memory = conversation_data.get('memory', [])
        
    def get_conversation_stats(self) -> Dict[str, Any]:
        """대화 통계"""
        user_messages = [msg for msg in self.history if msg['role'] == 'user']
        assistant_messages = [msg for msg in self.history if msg['role'] == 'assistant']
        
        return {
            'total_turns': len(self.history),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'memory_entries': len(getattr(self, 'memory', [])),
            'avg_user_message_length': sum(len(msg['content']) for msg in user_messages) / len(user_messages) if user_messages else 0,
            'avg_assistant_message_length': sum(len(msg['content']) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0
        }


class MultiUserChatBot:
    """다중 사용자 채팅봇"""
    
    def __init__(self, model, tokenizer, config: ChatConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ChatConfig()
        
        # 사용자별 채팅봇 인스턴스
        self.user_bots = {}
        
    def get_bot(self, user_id: str) -> ChatBot:
        """사용자별 채팅봇 가져오기"""
        if user_id not in self.user_bots:
            self.user_bots[user_id] = ChatBot(self.model, self.tokenizer, self.config)
        return self.user_bots[user_id]
        
    def chat(self, message: str, user_id: str, session_id: Optional[str] = None) -> str:
        """사용자별 채팅"""
        bot = self.get_bot(user_id)
        return bot.chat(message, user_id, session_id)
        
    def chat_stream(self, message: str, user_id: str, callback: Callable[[str], None], 
                   session_id: Optional[str] = None):
        """사용자별 스트리밍 채팅"""
        bot = self.get_bot(user_id)
        bot.chat_stream(message, callback, user_id, session_id)
        
    def clear_user_history(self, user_id: str):
        """특정 사용자 히스토리 초기화"""
        if user_id in self.user_bots:
            self.user_bots[user_id].clear_history()
            
    def get_all_users(self) -> List[str]:
        """모든 사용자 ID 반환"""
        return list(self.user_bots.keys())
        
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """사용자별 통계"""
        if user_id in self.user_bots:
            return self.user_bots[user_id].get_conversation_stats()
        return {}
