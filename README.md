# 🚀 LLM & RLHF Learning Journey (大语言模型与对齐算法实践)

本项目记录了我深入学习和实践大语言模型（LLM）核心架构，以及强化学习对齐算法（RLHF）的完整轨迹。

## 🧠 1. LLaMA3 架构解析与实践
深入剖析了 LLaMA3 的底层网络结构，并基于实际数据完成了前向推理与测试。
* **核心技术拆解**：深度理解并应用了 **GQA** (Grouped-Query Attention)、**RoPE** (旋转位置编码) 以及 **SwiGLU** 激活函数。
* **实践数据**：使用自定义/具体业务数据跑通了 LLaMA3 的核心流程，验证了模型结构在实际数据上的表现。

## ⚙️ 2. RLHF & PPO 算法复现
聚焦于大模型对齐技术，基于 DeepSpeed 框架从零跑通了基于人类反馈的强化学习流程。
* **核心目标**：学习并完整复现 RLHF (Reinforcement Learning from Human Feedback) 与 PPO (Proximal Policy Optimization) 算法。
* **学习资源与沉淀**：
    * 📖 **核心参考项目**：[lansinuote/Simple_RLHF](https://github.com/lansinuote/Simple_RLHF/blob/main/3.rlhf.ipynb)
    * 📝 **个人知识库**：[我的 RLHF 详细学习笔记 (Feishu)](https://my.feishu.cn/wiki/RvCiwFWHliRjpakcIuZcXBu1nEc)
    * 🗺️ **流程拆解图**：深入梳理的算法逻辑，详情见本仓库 [`RLHF流程拆解导图.png`](./RLHF流程拆解导图.png)