from embedding_store import EmbeddingStore
from bm25_retriever import BM25Retriever
from reranker import rerank
from config import settings
from logger import rag_logger
import httpx
import json
import traceback
import asyncio

# 全局实例
vector_store = EmbeddingStore()
bm25_store = BM25Retriever()


async def llm_stream_answer(prompt: str):
    """调用通义千问dashscope，一次性获取结果，模拟流式输出给前端"""
    api_key = settings.DASHSCOPE_API_KEY
    rag_logger.info(f"读取到的API_KEY前缀: {api_key[:12]}")
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": settings.LLM_MODEL,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {
            "result_format": "message",
            "stream": True
        }
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code != 200:
                err_text = resp.text
                rag_logger.error(f"大模型API调用失败，code:{resp.status_code},详情:{err_text}")
                yield "【大模型接口调用失败】"
                return
            data = resp.json()
            rag_logger.info(f"【LLM返回完整json】{data}")
            choice = data["output"]["choices"][0]
            content = choice["message"]["content"]
            # 模拟流式：逐字yield，前端依然是流式渲染效果
            for char in content:
                yield char
    except Exception as e:
        err_stack = traceback.format_exc()
        rag_logger.error(f"llm_stream_answer 异常:\n{err_stack}")
        yield "【调用大模型发生异常】"


async def rag_query(user_query: str):
    rag_logger.info(f"收到用户提问：{user_query}")
    vector_docs = []
    try:
      vector_docs = await vector_store.search(user_query, 3)
      rag_logger.info(f"向量检索召回 {len(vector_docs)} 条")
    except Exception as e:
     rag_logger.error(f"向量检索异常: {str(e)}")
     vector_docs = []

    # 2.BM25检索：同步函数放到线程池，Python3.8兼容，增加异常捕获
    bm25_docs = []
    try:
        # 判断BM25是否存在文档，避免空库报错
        if hasattr(bm25_store, "has_documents") and bm25_store.has_documents():
            bm25_docs = await asyncio.get_event_loop().run_in_executor(
                None, bm25_store.search, user_query, 3
            )
            rag_logger.info(f"BM25检索召回 {len(bm25_docs)} 条")
        else:
            rag_logger.warning("BM25库为空，跳过BM25检索")
    except Exception as e:
        rag_logger.warning(f"BM25检索异常，降级不使用BM25: {str(e)}")
        bm25_docs = []

    # 合并两路召回
    all_docs = vector_docs + bm25_docs
    rag_logger.info(f"【调试】合并后的all_docs: {all_docs}")

    # ========== 空召回判断 ==========
    if len(all_docs) == 0:
        rag_logger.warning("向量库、BM25均未召回文档，跳过重排，无参考上下文")
        context = ""
    else:
        # ✅ 如果rerank是async异步函数，直接await，不要run_in_executor
        try:
            relevant_docs = await rerank(user_query, all_docs, 3)
            rag_logger.info(f"【调试】重排后relevant_docs: {relevant_docs}")
            # ✅ 新增：重排结果为空，降级使用原始召回文档
            if not relevant_docs:
                rag_logger.warning("重排过滤后无结果，降级使用原始召回文档")
                relevant_docs = all_docs
            # 判断元素类型：字典取text，字符串直接使用
            text_list = []
            for doc in relevant_docs:
                if isinstance(doc, dict) and "text" in doc:
                    text_list.append(doc["text"])
                else:
                    text_list.append(str(doc))
            context = "\n".join(text_list)
        except Exception as e:
            rag_logger.warning(f"重排模块异常，直接使用原始召回结果: {str(e)}")
            # 重排失败降级，兼容两种格式：dict / string
            text_list = []
            for doc in all_docs[:3]:
                if isinstance(doc, dict) and "text" in doc:
                    text_list.append(doc["text"])
                else:
                    text_list.append(str(doc))
            context = "\n".join(text_list)
    # ====================================
    rag_logger.info(f"【调试】最终拼入Prompt的上下文：>>> {context} <<<")
    # ✅RAG提示词
    system_prompt = f"""
你是文档问答助手，只能基于下面参考内容回答问题。
参考文档：
{context}
如果文档里没有答案，直接回答：文档中没有相关信息，不要编造。
用户问题：{user_query}
"""
    # 调用LLM流式生成
    async for chunk in llm_stream_answer(system_prompt):
        yield chunk