# -*- coding: utf-8 -*-
"""
@desc: 构建并编译Agentic RAG的工作流图
"""

from langgraph.graph import StateGraph, END

from agentic_rag.state import AgentState
from agentic_rag.nodes import (
    retrieve_memory_node,
    consolidate_memory_node,
    route_query_node,
    rewrite_query_node,
    retrieve_documents_node,
    grade_documents_node,
    generate_response_node,
    grade_relevance_node,
    direct_response_node
)


def switch_strategy(state: AgentState) -> dict:
    """切换到下一个未尝试的检索策略，并更新状态"""
    tried = state.get("tried_routes", [])
    all_routes = ['hierarchical_search', 'direct_chunk_search', 'web_search']
    for route in all_routes:
        if route not in tried:
            tried.append(route)
            print(f"--- 切换到新策略: {route} ---")
            return {"route": route, "tried_routes": tried}
    # 所有策略均已尝试（兜底，实际上外面条件边已处理）
    return {"route": state["route"], "tried_routes": tried}


def build_graph():
    """构建并返回集成了“自省”能力的、包含内外双循环的LangGraph图。"""
    workflow = StateGraph(AgentState)

    # --- 添加所有节点 ---
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("consolidate_memory", consolidate_memory_node)
    workflow.add_node("route_query", route_query_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("direct_response", direct_response_node)
    workflow.add_node("grade_relevance", grade_relevance_node)
    # 新增：策略切换节点
    workflow.add_node("switch_strategy", switch_strategy)

    # --- 定义边 ---
    workflow.set_entry_point("retrieve_memory")
    workflow.add_edge("retrieve_memory", "route_query")

    # 路由后，需要检索的走重写，直接对话走直接回答
    workflow.add_conditional_edges(
        "route_query",
        lambda state: state["route"],
        {
            "web_search": "rewrite_query",
            "hierarchical_search": "rewrite_query",
            "direct_chunk_search": "rewrite_query",
            "direct": "direct_response"
        }
    )

    # 重写 -> 检索 -> 评估
    workflow.add_edge("rewrite_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")

    # 文档评估后的决策（内循环）
    def decide_after_document_grading(state: AgentState):
        if state.get("documents_are_relevant"):
            print("---决策：文档相关，进入答案生成---")
            return "generate"

        print("---决策：文档不相关，尝试切换策略---")
        tried = state.get("tried_routes", [])
        available = ['hierarchical_search', 'direct_chunk_search', 'web_search']
        for r in available:
            if r not in tried:
                return "switch_strategy"
        print("---决策：所有检索策略均失败，流程结束---")
        return "fallback"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_after_document_grading,
        {
            "generate": "generate_response",
            "switch_strategy": "switch_strategy",   # 去切换策略
            "fallback": END
        }
    )

    # 策略切换后重新检索
    workflow.add_edge("switch_strategy", "retrieve_documents")

    # 生成答案 / 直接回答 -> 评估相关性
    workflow.add_edge("generate_response", "grade_relevance")
    workflow.add_edge("direct_response", "grade_relevance")

    # 答案评估后的决策（外循环）
    def decide_after_answer_grading(state: AgentState):
        if state["is_relevant"]:
            print("---决策：答案相关，流程结束---")
            return "end"

        if state.get("correction_attempts", 0) >= 2:
            print("---决策：已达到最大重试次数，流程结束---")
            return "end"
        else:
            print("---决策：答案不相关，触发修正性重写---")
            return "retry"

    workflow.add_conditional_edges(
        "grade_relevance",
        decide_after_answer_grading,
        {
            "end": "consolidate_memory",
            "retry": "rewrite_query"
        }
    )

    workflow.add_edge("consolidate_memory", END)

    return workflow.compile()
