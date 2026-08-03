复现Search-R1的过程记录
# 学习记录

1. 2026.8.2 学习data_process

# 项目相关
## 一、整个项目的文件地图

Search-R1/
│
├── README.md                     # Search-R1 官方说明、安装与 Quick Start
├── VERL_README.md                # 原始 veRL 框架说明
├── LICENSE / Notice.txt          # 许可证和第三方声明
├── setup.py
├── pyproject.toml
├── requirements.txt              # Python 依赖
│
├── train_ppo.sh                  # PPO 实验入口
├── train_grpo.sh                 # GRPO 实验入口
├── retrieval_launch.sh           # 默认本地检索服务启动脚本
├── infer.py                      # 单问题“推理—搜索—推理”示例
│
├── search_r1/                    # Search-R1 自己最核心的代码
│   ├── __init__.py
│   │
│   ├── llm_agent/
│   │   ├── generation.py         # 多轮搜索 Agent / rollout 核心
│   │   └── tensor_helper.py      # padding、mask、position_ids 等张量操作
│   │
│   └── search/
│       ├── retrieval_server.py   # FastAPI 检索服务器：BM25 / Dense
│       ├── retrieval.py          # 独立、离线使用的检索器实现
│       ├── retrieval_request.py  # 测试 /retrieve API 的简单客户端
│       ├── retrieval.sh          # 检索脚本
│       │
│       ├── index_builder.py      # Dense corpus 向量化、构建 FAISS 索引
│       ├── build_index.sh        # 索引构建入口
│       │
│       ├── rerank_server.py      # CrossEncoder 重排服务
│       ├── retrieval_rerank_server.py
│       │                         # 检索 + rerank 合并服务
│       │
│       ├── google_search_server.py
│       │                         # Google 在线搜索后端
│       └── serp_search_server.py # SerpAPI 在线搜索后端
│
├── scripts/
│   ├── download.py / download.sh # 下载语料和索引
│   ├── upload.py / upload.sh     # 上传实验资源
│   │
│   ├── data_process/
│   │   ├── nq.py                 # 普通 NQ 数据预处理
│   │   ├── nq_rag.py             # RAG 形式的 NQ 数据
│   │   ├── nq_search.py          # Search-R1 形式的 NQ 数据，重点
│   │   ├── qa_search_train_merge.py
│   │   └── qa_search_test_merge.py
│   │                             # 多个 QA 数据集的训练/测试合并
│   │
│   └── nq_hotpotqa/
│       ├── README.md             # 七个 QA 数据集实验说明
│       ├── data_process.sh       # 批量数据处理
│       ├── evaluate.sh           # 评测脚本
│       ├── v0.1/
│       │   ├── train_ppo.sh
│       │   └── train_grpo.sh
│       ├── v0.2/
│       │   ├── train_ppo.sh
│       │   └── train_grpo.sh
│       └── v0.3/
│           ├── train_ppo_format.sh
│           └── train_grpo_format.sh
│
├── example/
│   ├── case.txt                  # 一条轨迹/输出示例
│   ├── corpus.jsonl              # 很小的示例语料库
│   │
│   ├── multinode/
│   │   ├── train_ppo_multinode_32b.sh
│   │   ├── train_grpo_multinode_32b.sh
│   │   └── train_grpo_multinode_72b.sh
│   │
│   └── retriever/
│       ├── retrieval_launch_ann.sh
│       ├── retrieval_launch_bm25.sh
│       ├── retrieval_launch_google.sh
│       ├── retrieval_launch_serpapi.sh
│       └── retrieval_launch_hierarchical.sh
│
├── docs/
│   ├── experiment_log.md         # v0.1/v0.2/v0.3 的演化记录
│   ├── retriever.md              # 检索后端说明
│   └── multinode.md              # 多机训练说明
│
├── public/                       # README 图片、项目展示资源
│
└── verl/                         # 内置并修改的 veRL 强化学习框架
    ├── protocol.py               # DataProto：模块之间的数据容器
    │
    ├── trainer/
    │   ├── main_ppo.py           # PPO/GRPO 主程序与 RewardManager
    │   ├── main_ppo_format.py    # 带格式奖励版本
    │   ├── main_eval.py
    │   ├── main_generation.py
    │   ├── fsdp_sft_trainer.py
    │   │
    │   ├── config/
    │   │   ├── ppo_trainer.yaml  # 全局训练配置，重点
    │   │   ├── evaluation.yaml
    │   │   ├── generation.yaml
    │   │   ├── sft_trainer.yaml
    │   │   └── ppo_megatron_trainer.yaml
    │   │
    │   └── ppo/
    │       ├── ray_trainer.py    # 整个 RL 数据流，最重要
    │       └── core_algos.py     # GAE、GRPO advantage、KL 等数学实现
    │
    ├── workers/
    │   ├── fsdp_workers.py       # FSDP actor/critic/ref worker
    │   ├── megatron_workers.py   # Megatron worker
    │   ├── actor/                # PPO actor 更新
    │   ├── critic/               # PPO value/critic
    │   ├── reward_model/
    │   ├── rollout/
    │   │   ├── hf_rollout.py
    │   │   └── vllm_rollout/
    │   │       └── vllm_rollout.py
    │   └── sharding_manager/     # FSDP/vLLM 权重和分片切换
    │
    ├── utils/
    │   ├── dataset/
    │   │   └── rl_dataset.py     # parquet → tokenizer → DataLoader
    │   ├── reward_score/
    │   │   ├── qa_em.py          # QA exact-match 奖励，重点
    │   │   └── qa_em_format.py   # 正确性 + 格式奖励
    │   ├── seqlen_balancing.py   # 多卡 token 数量负载均衡
    │   ├── torch_functional.py
    │   ├── fsdp_utils.py
    │   └── tracking.py           # WandB 等日志
    │
    ├── single_controller/        # Ray WorkerGroup 和远程调用抽象
    ├── models/                   # Llama/Transformers 模型适配
    └── third_party/vllm/         # vLLM 兼容层

## 二、真正的主调用链

整个项目可以压缩成这一条路径：

原始 QA 数据
    ↓
scripts/data_process/nq_search.py
    ↓
train.parquet / test.parquet
    ↓
train_grpo.sh 或 train_ppo.sh
    ↓
verl.trainer.main_ppo
    ↓
RewardManager + RayPPOTrainer
    ↓
LLMGenerationManager.run_llm_loop()
    ↓
模型生成 <search>query</search>
    ↓
POST /retrieve
    ↓
retrieval_server.py → BM25 / Dense FAISS
    ↓
返回 <information>...</information>
    ↓
模型继续思考、搜索，最后输出 <answer>
    ↓
qa_em.py 计算答案奖励
    ↓
GAE(PPO) 或 Group Advantage(GRPO)
    ↓
更新 Actor