import os
import faiss
import numpy as np
import httpx
from config import settings
from logger import rag_logger
from typing import List

class EmbeddingStore:
    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.model = settings.EMBEDDING_MODEL
        self.db_path = settings.VECTOR_DB_PATH
        self.index = None
        self.text_map: List[str] = []  # index id -> 原文片段
        self._load_index()

    async def get_embedding(self, text: str):
        """调用通义向量接口，返回向量数组"""
        url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "input": {"texts": [text]}
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, headers=headers, timeout=30)
            res = resp.json()
            vec = res["output"]["embeddings"][0]["embedding"]
        return np.array(vec, dtype=np.float32)

    def _load_index(self):
        """加载本地faiss向量库，如果不存在新建"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        if os.path.exists(self.db_path):
            self.index = faiss.read_index(self.db_path)
            rag_logger.info(f"加载已有向量库，向量数量：{self.index.ntotal}")
        else:
            self.index = faiss.IndexFlatL2(1536)
            rag_logger.info("新建空FAISS向量库")

    async def add_chunks(self, chunks: List[str]):
        """批量把文本片段转为向量存入faiss"""
        for chunk in chunks:
            vec = await self.get_embedding(chunk)
            self.index.add(vec.reshape(1,-1))
            self.text_map.append(chunk)
        # 持久化保存到本地文件
        faiss.write_index(self.index, self.db_path)
        rag_logger.info(f"{len(chunks)}个片段写入向量库并持久化")

    async def search(self, query: str, top_k: int = 3) -> List[str]:
        """向量检索，返回top-k原文片段"""
        q_vec = await self.get_embedding(query)
        dists, ids = self.index.search(q_vec.reshape(1,-1), top_k)
        results = []
        for idx in ids[0]:
            if idx >=0 and idx < len(self.text_map):
                results.append(self.text_map[idx])
        rag_logger.info(f"向量检索命中 {len(results)} 条片段")
        return results