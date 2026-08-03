# 学习LLM

本项目记录了我深入学习和实践大语言模型（LLM）核心架构，以及强化学习对齐算法（RLHF）的完整轨迹。

## 0. Transformer系列代码复现
参考李牧的Transformer讲解

## 1. LLaMA3 架构解析与实践
复现了 **GQA** (Grouped-Query Attention)、**RoPE** (旋转位置编码) 以及 **SwiGLU** 激活函数。跑了下LLama3
## 2. DeepSpeed - RLHF & PPO 算法复现
复现了DeepSpeed的RLHF的项目

**参考项目来源**：[lansinuote/Simple_RLHF](https://github.com/lansinuote/Simple_RLHF/blob/main/3.rlhf.ipynb)

**笔记**：[我的 RLHF 详细学习笔记 (Feishu)](https://my.feishu.cn/wiki/RvCiwFWHliRjpakcIuZcXBu1nEc)

**RLHF流程拆解图**：深入梳理的算法逻辑，详情见本仓库 [`RLHF流程拆解导图.png`](./RLHF流程拆解导图.png)

## 3. huggingface-transformer库熟悉

使用了transformers库里面的各种函数的使用， 只是稍微看了下， 大概知道有什么东西，基本看懂

**参考项目来源**：[zyds/transformers-code](https://github.com/zyds/transformers-code)

**视频教程**: [你可是处女座啊的视频 | 可以参考下，反正我就看了两个就没看了](https://www.bilibili.com/video/BV1KX4y1a7Jk)

后续复现参考官方文档 : [Transformers快速入门](https://transformers.run/)

## 04. 复现和研究DeepSeek
MOE部分：**参考资料** [懂点AI事 | 搭配视频感觉学会的更快](https://www.bilibili.com/video/BV1RtNLeqEeu/?vd_source=ec51181096be43428187c61347965a9b)


## 05. Langchain工具学习使用

学习Langchain的工具使用，参考资料
1. [dive2LangGraph](https://larryli93.github.io/dive-into-langgraph-plus/)
2. [dive2LangGraph | github代码](https://github.com/LarryLi93/dive-into-langgraph-plus/blob/main/.env.example)
3. [官方文档的中文翻译](https://langchain-doc.cn/)
4. [视频教程 | 我没看](https://www.bilibili.com/video/BV1pRiWB8EXy/?vd_source=ec51181096be43428187c61347965a9b)

2026.4.16 - 20264.19 :主要是学习基础的使用01-06 大概懂了， MCP没学感觉dive2LangGraph里面看不懂

## 06. RAG学习

第六部分是学习RAG，参考资料主要是：[MasteringRAG](https://github.com/Steven-Luo/MasteringRAG/tree/main?tab=readme-ov-file)

2026.4.19 - 2026.4.21 学习 01. 使用RAG技术构建企业级文档问答系统之QA抽取
感觉MasteringRAG对于新手的我有点难，后面去学07部分了

## 07. ALL in RAG

2026.4.22-2026.5.9 有点琐事，学的慢了点

第七部分也是学RAG，但是感觉这个项目应该更加适合新手

我直接使用git clone https://github.com/datawhalechina/all-in-rag.git 把项目clone到文件夹学

同时参考[ALL in RAG官方文档](https://datawhalechina.github.io/all-in-rag/#/chapter1/02_preparation?id=%e5%9b%9b%e3%80%81windows%e7%8e%af%e5%a2%83%e9%85%8d%e7%bd%ae%ef%bc%88%e4%bd%bf%e7%94%a8cloud-studio-%e6%88%96-codespaces-%e5%8f%af%e8%b7%b3%e8%bf%87%e6%ad%a4%e6%ad%a5%e9%aa%a4%ef%bc%89)

## 08. export-rag
 
 2026.5.9 - ？ 这里我自己企图做一个外贸客服的Agent

 ## 09. Agentic-RAG

2026.5.9 - 2026.5.10 复现了下Corrective RAG，知道了RAG和llm结合的几种方式

## Agent相关的开发先放一放，上训练

# 训练篇

## 10. 高效微调

[微调原理 | 视频没看，直接看了下面的代码](https://www.bilibili.com/video/BV1Xu4y1k7Ls/)

2026.5.18 - 2026.5.19 开始实战嬛嬛chat
[PEFT实战 | 嬛嬛chat](https://github.com/datawhalechina/self-llm/blob/master/examples/Chat-%E5%AC%9B%E5%AC%9B/readme.md)

## 11.高效微调 + LLamafactory使用|Anti-Frad
2026.5.20 - 2026.5.24 
开始学习Anti-frad，复现了Lora单卡训练
学习使用LLamafactory 一键式Lora微调，只要先配置好joson文件数据库注册，然后再配置好yaml训练参数脚本进行训练就行
[sft + peft实战 | anti-frad](https://github.com/golfxiao/anti_fraud_sft)

## 12. 吴恩达强化学习教程
2026.5.27 - 
[吴恩达强化学习课程，全看完了，不长](https://www.bilibili.com/video/BV1PmJzzpEqG/?vd_source=ec51181096be43428187c61347965a9b)

## 13. Hands on RL 项目学习
在github上了Hands on RL 具体的学习RL 和 Agentic RL 里面各种微调策略DPO，PPO，GRPO，想要敲代码实现
2026.6.30 - 2026.7.31
[Hands on RL 项目学习](https://walkinglabs.github.io/hands-on-modern-rl/preface/env-setup)

2026.7.24 学习GRPO
2026.7.25 学习DPO

2026.8.1 学习20章Agentic RL

## 14 Search-R1
2026.7.31 - 2026.8 学习Hands on RL 结合 Search-R1的知识进行学习
先把Search Agent放一放，弄一个Code Agent


## 15 Swe-RL
2026.8.3 Code Agent 开启


