"""
텍스트 전처리 모듈
"""

import re
import unicodedata
from typing import List, Optional, Callable
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class TextPreprocessor:
    """텍스트 전처리 클래스"""
    
    def __init__(self, 
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = False,
                 lowercase: bool = True,
                 normalize_unicode: bool = True,
                 min_length: int = 1,
                 max_length: Optional[int] = None,
                 custom_filters: Optional[List[Callable]] = None):
        
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.min_length = min_length
        self.max_length = max_length
        self.custom_filters = custom_filters or []
        
        # NLTK 리소스 초기화
        self._init_nltk()
        
    def _init_nltk(self):
        """NLTK 리소스 초기화"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess(self, text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
            
        # 유니코드 정규화
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
            
        # 소문자 변환
        if self.lowercase:
            text = text.lower()
            
        # 구두점 제거
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # 숫자 제거
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
            
        # 불용어 제거
        if self.remove_stopwords:
            words = word_tokenize(text)
            words = [word for word in words if word not in self.stop_words]
            text = ' '.join(words)
            
        # 길이 필터링
        words = text.split()
        if self.min_length and len(words) < self.min_length:
            return ""
        if self.max_length and len(words) > self.max_length:
            words = words[:self.max_length]
            text = ' '.join(words)
            
        # 커스텀 필터 적용
        for filter_func in self.custom_filters:
            text = filter_func(text)
            
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """배치 텍스트 전처리"""
        return [self.preprocess(text) for text in texts]
        
    def add_custom_filter(self, filter_func: Callable[[str], str]):
        """커스텀 필터 추가"""
        self.custom_filters.append(filter_func)
        
    def remove_emails(self, text: str) -> str:
        """이메일 주소 제거"""
        return re.sub(r'\S+@\S+', '', text)
        
    def remove_urls(self, text: str) -> str:
        """URL 제거"""
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
    def remove_mentions(self, text: str) -> str:
        """멘션 제거 (@username)"""
        return re.sub(r'@\w+', '', text)
        
    def remove_hashtags(self, text: str) -> str:
        """해시태그 제거 (#hashtag)"""
        return re.sub(r'#\w+', '', text)
        
    def clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        return re.sub(r'<[^>]+>', '', text)
        
    def normalize_whitespace(self, text: str) -> str:
        """공백 정규화"""
        return re.sub(r'\s+', ' ', text).strip()


class ConversationPreprocessor(TextPreprocessor):
    """대화 데이터 전처리 클래스"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speaker_pattern = re.compile(r'^[A-Za-z]+:', re.MULTILINE)
        
    def preprocess_conversation(self, conversation: str) -> List[str]:
        """대화를 발화 단위로 분할하고 전처리"""
        # 발화자 제거
        conversation = self.speaker_pattern.sub('', conversation)
        
        # 문장 단위로 분할
        sentences = re.split(r'[.!?]+', conversation)
        
        # 각 문장 전처리
        processed_sentences = []
        for sentence in sentences:
            processed = self.preprocess(sentence)
            if processed:
                processed_sentences.append(processed)
                
        return processed_sentences
        
    def create_turn_pairs(self, conversation: List[str]) -> List[tuple]:
        """대화를 (질문, 답변) 쌍으로 변환"""
        pairs = []
        for i in range(len(conversation) - 1):
            pairs.append((conversation[i], conversation[i + 1]))
        return pairs


class CodePreprocessor(TextPreprocessor):
    """코드 데이터 전처리 클래스"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remove_punctuation = False  # 코드에서는 구두점 유지
        
    def preprocess_code(self, code: str) -> str:
        """코드 전처리"""
        # 주석 제거
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # 연속된 공백 정규화
        code = re.sub(r'\s+', ' ', code)
        
        return code.strip()


def create_preprocessor(preprocessor_type: str = 'default', **kwargs):
    """전처리기 팩토리 함수"""
    if preprocessor_type == 'default':
        return TextPreprocessor(**kwargs)
    elif preprocessor_type == 'conversation':
        return ConversationPreprocessor(**kwargs)
    elif preprocessor_type == 'code':
        return CodePreprocessor(**kwargs)
    else:
        raise ValueError(f"Unsupported preprocessor type: {preprocessor_type}")
