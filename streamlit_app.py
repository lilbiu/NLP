# streamlit_app.py
# 运行命令: streamlit run streamlit_app.py
import os
from typing import List, Dict

import requests
import streamlit as st

# 可通过环境变量设置后端 API 地址
API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="医疗知识智能问答",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# ===================== 全局自定义样式 =====================
st.markdown("""
<style>
    /* 隐藏原生控件 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 主容器边距清零，实现全屏布局 */
    .main .block-container {
        padding-top: 0;
        padding-bottom: 0;
        padding-left: 0;
        padding-right: 0;
        max-width: 100%;
    }

    /* 侧边栏样式 - 极简版 */
    [data-testid="stSidebar"] {
        background-color: #fafbfc;
        border-right: 1px solid #eaecef;
        width: 240px !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }

    /* ===== 顶部标题栏 - 固定常驻 ===== */
    .page-header {
        background: #ffffff;
        border-bottom: 1px solid #e6f0ff;
        padding: 1rem 2.5rem;
        /* 核心：固定定位，始终悬浮在顶部 */
        position: fixed;
        top: 0;
        left: 240px;
        right: 0;
        z-index: 999;
    }
    .page-header h1 {
        margin: 0;
        font-size: 1.25rem;
        color: #1a56a8;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .page-header p {
        margin: 0.2rem 0 0 0;
        font-size: 0.8rem;
        color: #667a99;
    }

    /* 聊天主容器 - 独立滚动，预留顶部标题空间 */
    .chat-wrapper {
        height: 100vh;
        /* 上下预留空间，避开固定的顶部标题和底部输入框 */
        padding-top: 90px;
        padding-bottom: 90px;
        overflow-y: auto;
        max-width: 900px;
        margin: 0 auto;
        box-sizing: border-box;
    }

    /* 欢迎区域 - 医疗主题 */
    .welcome-area {
        text-align: center;
        padding: 3rem 0 2rem;
    }
    .welcome-title {
        font-size: 2rem;
        color: #223354;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .welcome-subtitle {
        color: #7a8aa6;
        font-size: 0.95rem;
        margin-bottom: 2.5rem;
    }

    /* 推荐问题标签 */
    .suggestion-tag {
        background: #ffffff;
        border: 1px solid #dce8f8;
        border-radius: 8px;
        padding: 0.75rem 1.2rem;
        font-size: 0.9rem;
        color: #2d5a9e;
        cursor: pointer;
        transition: all 0.2s;
        text-align: center;
    }
    .suggestion-tag:hover {
        background: #f5f9ff;
        border-color: #1a73e8;
    }

    /* 底部输入区域 */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 240px;
        right: 0;
        padding: 0.8rem 2.5rem 1.2rem;
        background: #fff;
        border-top: 1px solid #f0f0f0;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)


# ===================== 状态初始化 =====================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_url" not in st.session_state:
        st.session_state.api_url = API_URL


init_session_state()

# ===================== 左侧侧边栏（仅新对话） =====================
with st.sidebar:
    if st.button("✏️ 新对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""
    <div style="margin-top: 1rem; padding: 0 1rem; font-size: 0.8rem; color: #999;">
        点击开启全新对话<br>历史记录不会保留
    </div>
    """, unsafe_allow_html=True)

# ===================== 顶部常驻标题栏（对话中也不会消失） =====================
st.markdown("""
<div class="page-header">
    <h1>🏥 医疗知识智能问答系统</h1>
    <p>基于 LLM 和 RAG · 提供专业准确的医疗健康知识解答</p>
</div>
""", unsafe_allow_html=True)

# ===================== 聊天主区域 =====================
chat_wrapper = st.container()
with chat_wrapper:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

    if not st.session_state.messages:
        # 医疗主题欢迎界面
        st.markdown("""
        <div class="welcome-area">
            <div class="welcome-title">您好，请问有什么医疗健康问题？</div>
            <div class="welcome-subtitle">支持常见疾病、用药指导、健康养生、医学科普等问题咨询</div>
        </div>
        """, unsafe_allow_html=True)

        # 医疗相关推荐问题
        suggestions = [
            ["高血压患者日常有哪些注意事项？", "糖尿病饮食禁忌有哪些？", "感冒发烧如何正确用药？"],
            ["长期失眠怎么调理？", "高血脂吃什么食物好？", "颈椎病日常如何缓解？"],
            ["接种疫苗后有哪些注意事项？", "慢性胃炎如何养胃？", "干眼症怎么改善？"]
        ]

        for row in suggestions:
            cols = st.columns(3)
            for idx, question in enumerate(row):
                with cols[idx]:
                    if st.button(question, key=f"med_{question[:8]}_{idx}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "text": question})
                        st.rerun()
    else:
        # 渲染聊天消息
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])

    st.markdown('</div>', unsafe_allow_html=True)

# ===================== 底部输入区域（无工具栏） =====================
st.markdown('<div class="input-area">', unsafe_allow_html=True)
user_input = st.chat_input(placeholder="请输入您的医疗健康问题，按 Enter 发送")
st.markdown('</div>', unsafe_allow_html=True)

# ===================== 消息处理逻辑（保持原功能） =====================
if user_input and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "text": query})
    st.rerun()


def last_user_without_reply(messages: List[Dict]) -> bool:
    if not messages:
        return False
    return messages[-1]["role"] == "user"


if last_user_without_reply(st.session_state.messages):
    latest_query = st.session_state.messages[-1]["text"]
    try:
        with st.spinner("正在检索医疗知识库并生成解答..."):
            resp = requests.post(
                f"{st.session_state.api_url}/api/chat",
                json={"query": latest_query},
                timeout=300
            )
            resp.raise_for_status()
            data = resp.json()
            agent_text = data.get("response") or "(无回复内容)"
            st.session_state.messages.append({"role": "assistant", "text": agent_text})
    except requests.exceptions.ConnectionError:
        st.session_state.messages.append({
            "role": "assistant",
            "text": "❌ 连接失败：请确认后端服务已启动，且 API 地址配置正确"
        })
    except requests.exceptions.Timeout:
        st.session_state.messages.append({
            "role": "assistant",
            "text": "⏱️ 请求超时：后端处理时间过长，请稍后重试"
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "text": f"抱歉，您所说的不属于疾病范畴，小的只能回答有关医疗健康方面的问题哦~"
        })
    finally:
        st.rerun()
