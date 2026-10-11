import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

class Settings:
    # 增加默认值""，避免None
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen-turbo")
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "bge-reranker-v2-m3")
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_db/index.faiss")
    # 限流：每分钟最多5次请求
    RATE_LIMIT = "5/minute"

settings = Settings()