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


## 04. 复现和研究DeepSeek
MOE部分：**参考资料** [懂点AI事 | 搭配视频感觉学会的更快](https://www.bilibili.com/video/BV1RtNLeqEeu/?vd_source=ec51181096be43428187c61347965a9b)
