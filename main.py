from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from rag_pipeline import rag_query, vector_store, bm25_store
from document_loader import load_document
from text_splitter import split_text
from config import settings
from logger import rag_logger
import json
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="原生RAG项目｜混合检索+重排")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
"""app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# ========== 全局异常捕获中间件 ==========
@app.middleware("http")
async def catch_all_exceptions(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except json.JSONDecodeError:
        logger.warning(f"JSON解析失败，请求路径: {request.url.path}")
        return JSONResponse(
            status_code=400,
            content={"code": 400, "msg": "JSON格式错误，请检查请求体", "data": None}
        )
    except HTTPException as http_err:
        # 已经手动抛出的HTTP异常，直接透传
        return JSONResponse(
            status_code=http_err.status_code,
            content={"code": http_err.status_code, "msg": http_err.detail, "data": None}
        )
    except Exception as e:
        # 其他所有未知异常
        logger.error(f"服务内部异常: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "msg": "服务器内部错误", "data": None}
        )"""

# 前端聊天页面，内嵌html，新增加载演示文档按钮
html_page = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>RAG文档问答</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:"Microsoft Yahei",sans-serif;}
body{max-width:900px;margin:30px auto;padding:0 20px;}
h1{text-align:center;margin-bottom:30px;color:#333;}
.input-area{display:flex;gap:10px;margin-bottom:20px;}
#question{flex:1;padding:12px 14px;font-size:16px;border:1px solid #ccc;border-radius:8px;}
button{padding:12px 20px;border:none;border-radius:8px;background:#2563eb;color:white;font-size:15px;cursor:pointer;}
button:disabled{background:#94a3b8;cursor:not-allowed;}
#answer{min-height:200px;padding:16px;border:1px solid #e5e7eb;border-radius:8px;line-height:1.7;font-size:16px;white-space:pre-wrap;background:#fafafa;}
</style>
</head>
<body>
<h1>RAG文档问答</h1>
<div class="input-area">
<input type="text" id="question" placeholder="请输入你的问题">
<button id="loadBtn">加载测试文档</button>
<button id="sendBtn">发送</button>
</div>
<div id="answer"></div>

<script>
const questionInput = document.getElementById('question');
const answerDom = document.getElementById('answer');
const loadBtn = document.getElementById('loadBtn');
const sendBtn = document.getElementById('sendBtn');

loadBtn.onclick = async function(){
  const res = await fetch("/load_demo",{method:"POST"});
  const data = await res.json();
  alert(data.msg);
}

sendBtn.onclick = async function(){
  const q = questionInput.value.trim();
  if(!q) return;
  answerDom.innerText = "";
  sendBtn.disabled = true;

  const resp = await fetch("/rag/query",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({query:q})
  });
  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  while(true){
    const {done,value} = await reader.read();
    if(done) break;
    buf += decoder.decode(value,{stream:true});
    const parts = buf.split("\\n\\n");
    buf = parts.pop();
    for(let p of parts){
      p = p.trim();
      if(p.startsWith("data:")){
        const jsonStr = p.slice(5).trim();
        try{
          const obj = JSON.parse(jsonStr);
          answerDom.innerText += obj.content;
        }catch(e){}
      }
    }
  }
  sendBtn.disabled = false;
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return html_page

class QueryRequest(BaseModel):
    query: str

@app.post("/rag/query")
async def rag_query_api(query: QueryRequest):
    async def stream_generator():
        async for chunk in rag_query(query.query):
            yield f"data:{json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

@app.post("/load_doc")
async def load_doc_api(request: Request):
    """接口：加载本地文档，存入向量库+BM25"""
    body = await request.json()
    file_path = body["file_path"]
    # 清空旧的BM25数据，避免重复
    bm25_store.clear()
    text = load_document(file_path)
    chunks = split_text(text)
    await vector_store.add_chunks(chunks)
    bm25_store.add_texts(chunks)
    return {"msg":"文档加载完成", "chunk_count": len(chunks)}

@app.post("/load_demo")
async def load_demo_text():
    """加载内置演示文本，不需要本地文件，快速测试RAG链路"""
    demo_text = """
RAG全称检索增强生成（Retrieval-Augmented Generation）。
RAG的作用：在大模型回答之前，先检索外部知识库，把相关文档塞进prompt，减少大模型幻觉。
RAG分为几个阶段：文档加载、文本切分、向量化存入向量库、用户提问、检索、重排、组装上下文、大模型生成答案。
混合检索 = BM25关键词检索 + FAISS向量语义检索。
BM25：关键词匹配，适合专有名词、精确术语。
向量检索：语义匹配，意思相近就能找到。
Rerank重排：对检索出来的文档重新打分排序，过滤无关文档。
"""
    # 清空旧BM25，防止多次加载重复叠加文本
    bm25_store.clear()
    chunks = split_text(demo_text)
    await vector_store.add_chunks(chunks)
    bm25_store.add_texts(chunks)
    rag_logger.info("✅ 内置演示文档加载完成")
    return {"msg":"内置测试文档加载成功，可以开始提问！", "chunk_count": len(chunks)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)