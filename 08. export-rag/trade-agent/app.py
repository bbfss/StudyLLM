from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.agent import TradeAgent


ROOT = Path(__file__).parent


@st.cache_resource
def get_agent() -> TradeAgent:
    return TradeAgent(data_dir=ROOT / "data", db_path=ROOT / "runtime" / "customer_memory.db")


st.set_page_config(page_title="Trade Agent Demo", page_icon="AI", layout="wide")
st.title("外贸谈判客服 Agent")
st.caption("RAG + 意图识别 + 谈判策略 + SQLite 客户记忆")

with st.sidebar:
    st.subheader("Demo 输入")
    customer_id = st.text_input("Customer ID", value="demo_buyer")
    preset = st.selectbox(
        "选择测试消息",
        [
            "Can you give me a better price if I order 500 pieces?",
            "How long does shipping to the US take?",
            "Can I get a sample before bulk order?",
            "We want custom packaging for 300 pieces. Is it possible?",
            "Some units are broken. Can you refund us?",
        ],
    )

message = st.text_area("客户消息", value=preset, height=120)

if st.button("Run Agent", type="primary"):
    result = get_agent().run(message, customer_id=customer_id)
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("结构化输出")
        st.json(
            {
                "intent": result["intent"],
                "customer_need": result["customer_need"],
                "negotiation_strategy": result["negotiation_strategy"],
                "suggested_reply": result["suggested_reply"],
                "next_action": result["next_action"],
                "customer_profile": result["customer_profile"],
            }
        )
    with right:
        st.subheader("检索依据")
        for item in result["retrieved_policy"]:
            st.markdown(f"**{item['source']} - {item['title']}**")
            st.write(item["content"])
            st.caption(f"score: {item['score']}")

    st.subheader("执行日志")
    for step in result["execution_log"]:
        st.write(f"- {step}")
else:
    st.info("点击 Run Agent 查看意图识别、RAG 检索、谈判策略和客户记忆。")
