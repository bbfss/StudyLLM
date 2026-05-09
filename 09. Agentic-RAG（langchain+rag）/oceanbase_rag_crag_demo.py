import argparse
import os
from pathlib import Path
from typing import List

import pymysql
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_oceanbase.vectorstores import OceanbaseVectorStore
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def get_connection_args() -> dict:
    return {
        "host": os.getenv("OB_HOST", "127.0.0.1"),
        "port": int(os.getenv("OB_PORT", "2881")),
        "user": os.getenv("OB_USER", "root@test"),
        "password": os.getenv("OB_PASSWORD", ""),
        "db_name": os.getenv("OB_DATABASE", "test"),
    }


def check_db() -> None:
    args = get_connection_args()
    conn = pymysql.connect(
        host=args["host"],
        port=args["port"],
        user=args["user"],
        password=args["password"],
        database=args["db_name"],
        charset="utf8mb4",
        connect_timeout=10,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            print("OceanBase connection ok:", cursor.fetchone())
    finally:
        conn.close()


def load_documents(data_path: str) -> List[Document]:
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Data path does not exist: {root}")

    docs: List[Document] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(path), "file_name": path.name},
                )
            )
        elif suffix == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            docs.extend(PyPDFLoader(str(path)).load())

    if not docs:
        raise ValueError(f"No .md, .txt or .pdf files found in: {root}")

    print(f"Loaded documents: {len(docs)}")
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split chunks: {len(chunks)}")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List[Document], drop_old: bool) -> OceanbaseVectorStore:
    vector_store = OceanbaseVectorStore(
        embedding_function=get_embeddings(),
        table_name=os.getenv("OB_TABLE_NAME", "langchain_knowledge_base"),
        connection_args=get_connection_args(),
        vidx_metric_type="l2",
        index_type="HNSW",
        drop_old=drop_old,
        normalize=True,
    )

    if chunks:
        ids = vector_store.add_documents(chunks)
        print(f"Added documents: {len(ids)}")

    return vector_store


def get_llm(temperature: float = 0) -> ChatOpenAI:
    api_key = os.getenv("AIHUBMIX_API_KEY")
    base_url = os.getenv("AIHUBMIX_BASE_URL")
    model = os.getenv("AIHUBMIX_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError("Please set AIHUBMIX_API_KEY in .env")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


def format_docs(docs: List[Document], max_chars: int = 3500) -> str:
    parts = []
    total = 0
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        text = f"[Document {i}] source={source}\n{doc.page_content}\n"
        if total + len(text) > max_chars:
            break
        parts.append(text)
        total += len(text)
    return "\n".join(parts)


def generate_answer(question: str, docs: List[Document]) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """你是一个严谨的 RAG 问答助手。
请只根据给定上下文回答用户问题。
如果上下文不足，请直接说“资料不足，无法根据知识库回答”。

用户问题：
{question}

上下文：
{context}

回答："""
    )
    chain = prompt | llm
    result = chain.invoke({"question": question, "context": format_docs(docs)})
    return result.content


def grade_documents(question: str, docs: List[Document]) -> dict:
    llm = get_llm(temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """你是一个文档相关性评分器。
请判断“检索到的文档”是否能帮助回答“用户问题”。

只返回 JSON，不要返回多余解释：
{{"relevant": true 或 false, "reason": "简短理由"}}

用户问题：
{question}

检索到的文档：
{context}
"""
    )
    result = (prompt | llm).invoke(
        {"question": question, "context": format_docs(docs, max_chars=2500)}
    )

    import json
    import re

    text = result.content.strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"relevant": False, "reason": f"评分器没有返回 JSON：{text[:100]}"}

    try:
        data = json.loads(match.group(0))
        return {
            "relevant": bool(data.get("relevant")),
            "reason": str(data.get("reason", "")),
        }
    except Exception as exc:
        return {"relevant": False, "reason": f"JSON 解析失败：{exc}"}


def fallback_answer(question: str) -> str:
    return (
        "当前知识库检索结果与问题不相关，已触发托底。\n\n"
        f"你的问题是：{question}\n\n"
        "新手版暂时不接入真实 Web Search，因此这里先返回安全提示："
        "请补充相关资料到知识库，或者接入 Tavily/Bing Search 作为外部搜索工具。"
    )


def prepare_vector_store(data_path: str, drop_old: bool) -> OceanbaseVectorStore:
    docs = load_documents(data_path)
    chunks = split_documents(docs)
    return build_vector_store(chunks, drop_old=drop_old)


def run_basic(data_path: str, question: str, drop_old: bool) -> None:
    vector_store = prepare_vector_store(data_path, drop_old=drop_old)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    print(generate_answer(question, docs))


def run_crag(data_path: str, question: str, drop_old: bool) -> None:
    vector_store = prepare_vector_store(data_path, drop_old=drop_old)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    grade = grade_documents(question, docs)

    print(f"文档相关性：{grade['relevant']}")
    print(f"评分理由：{grade['reason']}")
    print()

    if grade["relevant"]:
        print(generate_answer(question, docs))
    else:
        print(fallback_answer(question))


def run_agent(data_path: str, question: str, drop_old: bool) -> None:
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:
        raise ImportError("Please install langgraph first: pip install langgraph") from exc

    vector_store = prepare_vector_store(data_path, drop_old=drop_old)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    @tool
    def search_knowledge_base(query: str) -> str:
        """从本地知识库中检索与问题相关的内容。"""
        return format_docs(retriever.invoke(query))

    agent = create_react_agent(
        model=get_llm(),
        tools=[search_knowledge_base],
    )
    result = agent.invoke({"messages": [("user", question)]})
    print(result["messages"][-1].content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check-db", "basic", "agent", "crag"], default="basic")
    parser.add_argument("--data", default="./data/C8/cook")
    parser.add_argument("--question", default="宫保鸡丁怎么做？")
    parser.add_argument(
        "--reuse-table",
        action="store_true",
        help="Reuse existing OceanBase vector table instead of dropping/rebuilding it.",
    )
    args = parser.parse_args()

    if args.mode == "check-db":
        check_db()
        return

    drop_old = not args.reuse_table
    if args.mode == "basic":
        run_basic(args.data, args.question, drop_old)
    elif args.mode == "agent":
        run_agent(args.data, args.question, drop_old)
    elif args.mode == "crag":
        run_crag(args.data, args.question, drop_old)


if __name__ == "__main__":
    main()
