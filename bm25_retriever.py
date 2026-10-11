import jieba
from rank_bm25 import BM25Okapi
from logger import rag_logger
from typing import List

class BM25Retriever:
    def __init__(self):
        self.corpus: List[str] = []
        self.tokenized_corpus = []
        self.bm25 = None

    def clear(self):
        """清空全部文档，重新加载"""
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        rag_logger.info("BM25检索库已清空")

    def add_texts(self, chunks: List[str]):
        self.corpus.extend(chunks)
        # jieba中文分词
        tokenized = [jieba.lcut(doc) for doc in chunks]
        self.tokenized_corpus.extend(tokenized)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        rag_logger.info(f"BM25加载 {len(chunks)} 个文本块")

    def search(self, query: str, top_k: int =3) -> List[str]:
        """关键词检索 BM25"""
        if self.bm25 is None:
            rag_logger.warning("BM25库为空，请先加载文档！")
            return []
        tokens = jieba.lcut(query)
        scores = self.bm25.get_scores(tokens)
        # 排序取topk
        idx_list = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        res = [self.corpus[i] for i in idx_list]
        rag_logger.info(f"BM25检索命中 {len(res)}")
        return res