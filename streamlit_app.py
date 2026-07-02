# streamlit_app.py
# streamlit run streamlit_app.py
import streamlit as st
import requests
import os
from typing import List, Dict

# 可通过环境变量设置后端 API 地址，例如：
# export AGENT_API_URL="http://localhost:8000"
API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Agentic RAG 聊天", layout="wide")

st.title("Agentic RAG — 聊天界面 (Streamlit)")
st.markdown("本页面通过 REST API 与本地运行的 Agentic RAG 后端通信。")

if "messages" not in st.session_state:
    st.session_state.messages = []  # each item: {"role": "user"/"agent", "text": ...}

col1, col2 = st.columns([3, 1])

with col1:
    chat_box = st.container()
    with chat_box:
        for m in st.session_state.messages:
            if m["role"] == "user":
                st.markdown(f"**你:** {m['text']}")
            else:
                st.markdown(f"**Agent:** {m['text']}")

    user_input = st.text_area("输入你的问题或指令（支持 !show_memories / !forget ...）", height=120)
    send = st.button("发送")

with col2:
    st.subheader("控制")
    if st.button("清空对话"):
        st.session_state.messages = []
    if st.button("查看记忆"):
        try:
            r = requests.get(f"{API_URL}/api/memories")
            r.raise_for_status()
            mems = r.json().get("memories", [])
            st.write(mems)
        except Exception as e:
            st.error(f"调用 /api/memories 失败: {e}")

    st.markdown("---")
    st.markdown("环境设置")
    st.text_input("API 地址 (AGENT_API_URL)", value=API_URL, key="api_url_display")

# 发送消息处理
if send and user_input and user_input.strip():
    query = user_input.strip()
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "text": query})
    # 清输入框（Streamlit 需要通过重新渲染或 JS，这里用清空变量）
    st.rerun()

# 当页面加载时，如果上次发送了消息并被存储，尝试发送到后端
# 这里通过检测最新一条是否为 user 且未被 agent 回复来触发请求
def last_user_without_reply(messages: List[Dict]) -> bool:
    if not messages:
        return False
    if messages[-1]["role"] != "user":
        return False
    # 若最后两条是 user->agent 则说明已回复
    if len(messages) >= 2 and messages[-2]["role"] == "agent":
        return False
    return True

if last_user_without_reply(st.session_state.messages):
    latest = st.session_state.messages[-1]["text"]
    try:
        with st.spinner("Agent 正在思考..."):
            resp = requests.post(f"{API_URL}/api/chat", json={"query": latest}, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            agent_text = data.get("response") or "(无回复)"
            # Append agent reply
            st.session_state.messages.append({"role": "agent", "text": agent_text})
            st.rerun()
    except Exception as e:
        st.session_state.messages.append({"role": "agent", "text": f"调用后端失败: {e}"})
        st.rerun()
