# api.py
import asyncio
import threading
import traceback
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agentic_rag import memory
from agentic_rag.graph import build_graph

app = FastAPI(title="Agentic RAG API")

# 简单 CORS，便于本地开发；部署时请按需收紧 allow_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------- 数据模型 -------------
class ChatRequest(BaseModel):
    query: str


class ForgetRequest(BaseModel):
    topic: str
    confirm: Optional[bool] = False


class ChatResponse(BaseModel):
    response: Optional[str]
    status: str
    detail: Optional[Any] = None


# ------------- 全局初始化 -------------
_graph = None
_graph_lock = threading.Lock()
_thread_id = str(uuid.uuid4())
_config = {"configurable": {"thread_id": _thread_id}}


@app.on_event("startup")
async def startup_event():
    """
    启动时：
      - 初始化长期记忆库（SQLite + Chroma collection）
      - 构建 LangGraph 工作流图（可能会初始化 LLM 客户端等）
    """
    global _graph
    # 初始化长期记忆库
    memory.initialize_memory_db()

    # 构建 LangGraph 图（阻塞且可能会初始化模型客户端）
    # 放在 startup 中同步执行，确保后续请求能立即使用 graph
    _graph = build_graph()


# def _sync_invoke_graph(query: str) -> Dict[str, Any]:
#     """
#     在工作线程中同步调用图并返回最终状态字典。
#     这个函数会被 run_in_executor 调用，以避免阻塞事件循环。
#     """
#     global _graph, _config
#     if _graph is None:
#         raise RuntimeError("Graph 未初始化")
#     try:
#         with _graph_lock:
#             inputs = {"query": query}
#             graph_config = {"recursion_limit": 10, **_config}
#             final_state = _graph.invoke(inputs, config=graph_config)
#             # final_state 可能是 dict 或自定义对象，这里以稳健方式提取 response
#             if isinstance(final_state, dict):
#                 response = final_state.get("response")
#             else:
#                 response = getattr(final_state, "response", None)
#             return {"response": response, "raw_state": final_state}
#     except Exception as e:
#         tb = traceback.format_exc()
#         raise RuntimeError(f"Graph invocation failed: {e}\n{tb}")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _sync_invoke_graph(query: str) -> Dict[str, Any]:
    global _graph, _config
    if _graph is None:
        raise RuntimeError("Graph 未初始化")
    try:
        with _graph_lock:
            inputs = {"query": query}
            graph_config = {"recursion_limit": 10, **_config}
            final_state = _graph.invoke(inputs, config=graph_config)
            if isinstance(final_state, dict):
                response = final_state.get("response")
            else:
                response = getattr(final_state, "response", None)
            return {"response": response, "raw_state": final_state}
    except Exception as e:
        logger.error("Graph invocation failed", exc_info=True)  # 打印完整堆栈
        tb = traceback.format_exc()
        raise RuntimeError(f"Graph invocation failed: {e}\n{tb}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    聊天接口：接受文本查询，返回 Agent 生成的答案。
    - 支持 CLI 中的记忆命令：!show_memories / !forget
    - 正常查询在后台线程里执行阻塞的 graph.invoke
    """
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    # 内置记忆命令
    if query == "!show_memories":
        mems = memory.view_memories(limit=20)
        return ChatResponse(response=None, status="ok", detail={"memories": mems})

    if query.startswith("!forget"):
        topic = query.replace("!forget", "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="!forget 命令需要跟主题，例如: !forget 我的项目ID")
        retrieved = memory.retrieve_memories(topic, top_k=5)
        deleted_ids = []
        for mem in retrieved:
            try:
                memory.delete_memory(mem["id"])
                deleted_ids.append(mem["id"])
            except Exception:
                continue
        return ChatResponse(response=None, status="ok", detail={"deleted_ids": deleted_ids})

    # 正常查询：在线程池中运行阻塞的图
    loop = asyncio.get_event_loop()
    try:
        final = await loop.run_in_executor(None, _sync_invoke_graph, query)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(response=final.get("response"), status="ok", detail=None)


@app.get("/api/memories")
async def list_memories(limit: int = 20):
    """返回最近的长期记忆条目。"""
    try:
        mems = memory.view_memories(limit=limit)
        return {"status": "ok", "memories": mems}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/forget")
async def forget_endpoint(req: ForgetRequest):
    """
    根据主题删除相关记忆；如果 confirm=False，则仅返回将被删除的候选项。
    """
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic 不能为空")
    try:
        candidates = memory.retrieve_memories(topic, top_k=10)
        if not req.confirm:
            # 返回候选项供客户端确认
            return {"status": "confirm", "candidates": candidates}

        deleted = []
        for mem in candidates:
            try:
                memory.delete_memory(mem["id"])
                deleted.append(mem["id"])
            except Exception:
                continue
        return {"status": "ok", "deleted_ids": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def status():
    return {"status": "ok", "thread_id": _thread_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000)
