# OceanBase + LangChain RAG/CRAG 新手本地跑通版

这份教程把两篇 OceanBase 文章合成一条更适合新手的路线：

- 原文 1：[LangChain V1 x OceanBase, Agentic RAG 实战 第二章](https://open.oceanbase.com/blog/25216154928)
- 原文 2：[Corrective RAG 详解 第三章](https://open.oceanbase.com/blog/25216184880)

目标不是一次性吃完所有概念，而是按这个顺序跑通：

```text
准备环境
-> 启动 OceanBase
-> 加载文档
-> 文档切块
-> 向量化
-> 写入 OceanBase
-> 基础 RAG 问答
-> Agentic RAG
-> Corrective RAG：文档评分 + 托底
```

## 0. 你要先理解这三个东西

**基础 RAG**

```text
用户问题 -> 检索知识库 -> 把检索结果塞给大模型 -> 生成回答
```

这是最稳定、最容易理解的版本。

**Agentic RAG**

```text
用户问题 -> Agent 判断要不要调用检索工具 -> 调用工具 -> 回答
```

它比基础 RAG 灵活，因为检索变成了 Agent 的一个工具。

**Corrective RAG**

```text
用户问题 -> 检索知识库 -> 判断检索结果是否相关
相关：正常回答
不相关：触发托底，比如 Web Search 或提示资料不足
```

它的核心是“不要盲信检索结果”。

## 1. 准备 Python 环境

建议新建一个单独环境，避免和你之前的 notebook 依赖混在一起。

```powershell
cd "E:\Code\StudyLLM\07. ALLinRag"
python -m venv .venv_ob
.\.venv_ob\Scripts\Activate.ps1
python -m pip install -U pip
```

安装依赖：

```powershell
pip install langchain langchain-openai langchain-community langchain-oceanbase
pip install langchain-text-splitters pypdf pymysql python-dotenv
pip install sentence-transformers langchain-huggingface pyobvector
```

如果你要跑 Agent 版本，再装：

```powershell
pip install langgraph
```

## 2. 启动 OceanBase

先确认 Docker Desktop 已经启动。

本目录已经准备了一个单独的 OceanBase compose 文件：

```powershell
docker compose -f docker-compose.oceanbase.yml up -d
```

第一次启动会拉镜像，比较慢。启动后等 1-3 分钟，再检查容器状态：

```powershell
docker ps
```

你应该能看到类似：

```text
oceanbase-rag
0.0.0.0:2881->2881/tcp
```

## 3. 配置 `.env`

在 `E:\Code\StudyLLM\07. ALLinRag\.env` 里补充这些配置：

```env
AIHUBMIX_API_KEY=你的key
AIHUBMIX_BASE_URL=你的base_url
AIHUBMIX_MODEL=你的模型名

OB_HOST=127.0.0.1
OB_PORT=2881
OB_USER=root@test
OB_PASSWORD=
OB_DATABASE=test
```

说明：

- Docker mini/slim 模式通常用 `root@test`。
- 本地测试密码通常为空。
- 如果你用的是云端 seekdb，就把 `OB_HOST`、`OB_PORT`、`OB_USER`、`OB_PASSWORD`、`OB_DATABASE` 换成控制台给你的连接信息。

## 4. 第一阶段：基础 RAG

基础 RAG 的代码已经放在：

```text
oceanbase_rag_crag_demo.py
```

运行：

```powershell
python oceanbase_rag_crag_demo.py --mode basic --data "./data/C8/cook" --question "宫保鸡丁怎么做？"
```

程序会做这些事：

```text
读取 Markdown/TXT/PDF
-> 切成小块
-> 用 HuggingFace Embedding 模型向量化
-> 写入 OceanBase 向量表，默认使用 L2 距离和 HNSW 索引
-> 检索 top-k 文档
-> 用 AIHubMix 大模型回答
```

最核心的链路长这样：

```python
docs = load_documents(data_path)
chunks = split_documents(docs)
vector_store = build_vector_store(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
context_docs = retriever.invoke(question)
answer = generate_answer(question, context_docs)
```

## 5. 第二阶段：Agentic RAG

基础 RAG 是“每次都检索”。Agentic RAG 是“把检索包装成工具，让 Agent 自己决定是否调用”。

运行：

```powershell
python oceanbase_rag_crag_demo.py --mode agent --data "./data/C8/cook" --question "推荐几道简单的素菜"
```

核心变化是多了一个工具：

```python
@tool
def search_knowledge_base(query: str) -> str:
    """从本地菜谱知识库中检索相关内容。"""
    docs = retriever.invoke(query)
    return format_docs(docs)
```

然后创建 Agent：

```python
agent = create_react_agent(
    model=llm,
    tools=[search_knowledge_base],
)
```

这个阶段你只要记住一句话：

```text
Agentic RAG = 把 RAG 检索能力做成 Agent 的工具。
```

## 6. 第三阶段：Corrective RAG

基础 RAG 有一个问题：即使检索结果不相关，它也可能硬答。

CRAG 增加一个评分步骤：

```text
先检索
-> 让大模型判断“检索结果和问题是否相关”
-> 相关：正常生成
-> 不相关：触发托底
```

运行：

```powershell
python oceanbase_rag_crag_demo.py --mode crag --data "./data/C8/cook" --question "2024年Nike第三季度收入是多少？"
```

这个问题和菜谱知识库明显不相关，所以会触发托底。

新手版先不直接写复杂 Middleware，而是用普通函数表达同样思想：

```python
docs = retriever.invoke(question)
grade = grade_documents(question, docs)

if grade["relevant"]:
    answer = generate_answer(question, docs)
else:
    answer = fallback_answer(question)
```

原文里的 Middleware 版本可以理解为：

```text
把 grade_documents 和 fallback_answer 插入 Agent 执行流程中。
```

先跑通函数版，再学 Middleware，会舒服很多。

## 7. 四个检查点

### 检查点 1：数据库连接

运行：

```powershell
python oceanbase_rag_crag_demo.py --mode check-db
```

看到类似输出就说明连接成功：

```text
OceanBase connection ok
```

### 检查点 2：向量写入

运行 basic 模式时，如果看到：

```text
Added documents: xxx
```

说明文档已经写入 OceanBase。

### 检查点 3：基础 RAG

问知识库内的问题：

```powershell
python oceanbase_rag_crag_demo.py --mode basic --data "./data/C8/cook" --question "土豆丝怎么做？"
```

如果回答能引用菜谱内容，说明基础 RAG 跑通。

### 检查点 4：CRAG 托底

问明显不属于菜谱知识库的问题：

```powershell
python oceanbase_rag_crag_demo.py --mode crag --data "./data/C8/cook" --question "OceanBase 向量索引有哪些类型？"
```

如果输出里出现：

```text
文档相关性：False
```

说明评分器起作用了。

## 8. 常见问题

**1. `ModuleNotFoundError: langchain_oceanbase`**

执行：

```powershell
pip install langchain-oceanbase
```

**2. 连接 OceanBase 失败**

先检查容器：

```powershell
docker ps
```

再确认 `.env`：

```env
OB_HOST=127.0.0.1
OB_PORT=2881
OB_USER=root@test
OB_PASSWORD=
OB_DATABASE=test
```

**3. Embedding 模型下载很慢**

第一次运行会下载模型。你可以先用较小模型：

```env
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

如果机器有 GPU，再考虑：

```env
EMBEDDING_MODEL=BAAI/bge-m3
```

**4. `create_react_agent` 报错**

先确认安装：

```powershell
pip install langgraph
```

如果还是报错，先跑：

```powershell
python oceanbase_rag_crag_demo.py --mode basic
python oceanbase_rag_crag_demo.py --mode crag
```

Agentic RAG 是进阶，不影响你理解主流程。

## 9. 学习顺序建议

不要一开始就看 Middleware。推荐顺序：

```text
1. 跑通 basic
2. 理解 load -> split -> embed -> store -> retrieve -> generate
3. 跑通 crag
4. 理解“评分 + 托底”
5. 最后再看 Agent 和 Middleware
```

你真正要掌握的不是某个库的写法，而是这条主线：

```text
RAG 解决“模型不知道你的资料”
Agentic RAG 解决“什么时候该查资料”
CRAG 解决“查到的资料靠不靠谱”
```
