from typing import List
import httpx
import logging
from dotenv import load_dotenv
import os

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
logger = logging.getLogger("rag-system")

async def rerank(query: str, docs: List[str], top_n: int = 3) -> List[str]:
    """重排文档，返回top_n最相关文档"""
    if not docs:
        logger.warning("重排输入文档为空，直接返回空列表")
        return []
    
    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "rerank-v1",
        "query": query,
        "documents": docs,
        "top_n": top_n
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        data = resp.json()
    # 安全读取，防止key不存在报错
    output = data.get("output", {})
    results = output.get("results", [])

    sorted_docs = []
    for item in results:
        idx = item.get("index")
        if idx is not None and 0 <= idx < len(docs):
            sorted_docs.append(docs[idx])
    return sorted_docs
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers, timeout=30)
        data = resp.json()
    out = []
    for item in data["output"]["results"]:
        out.append(item["document"])
    rag_logger.info(f"重排完成，筛选出 {len(out)} 条最相关片段")
    return out