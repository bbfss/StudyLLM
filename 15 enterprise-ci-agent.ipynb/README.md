┌─────────────────────────────────────────────────────────────┐
│ Step 1: Base model 选择                                     │
│   - 通常是 code-tuned LLM（如 Qwen-Coder、DeepSeek-Coder）  │
│   - 已经在大量代码上预训练                                  │
├─────────────────────────────────────────────────────────────┤
│ Step 2: SFT 冷启动（可选）                                  │
│   - 用 SWE-bench / SWE-smith 数据做 SFT                    │
│   - 让模型学会基本的 trajectory 格式                        │
├─────────────────────────────────────────────────────────────┤
│ Step 3: RL 训练                                             │
│   - GRPO / PPO                                              │
│   - Reward: 测试通过 binary                                │
│   - 长 horizon：每个 trajectory 可能 16-100+ 步             │
├─────────────────────────────────────────────────────────────┤
│ Step 4: Rejection sampling + 二次 SFT                      │
│   - 从 RL 训练后的模型生成多个候选                          │
│   - 选最好的做 SFT                                          │
├─────────────────────────────────────────────────────────────┤
│ Step 5: 评测                                                │
│   - SWE-bench Verified                                     │
│   - 内部 evaluation set                                    │
└─────────────────────────────────────────────────────────────┘


目前文件地图

enterprise-ci-agent/

├── notebooks/
│   └── lesson01_agent_loop.ipynb
│
├── src/
│   └── ci_agent/
│       ├── __init__.py
│       ├── agent.py
│       ├── environment.py
│       ├── model.py
│       └── trajectory.py
│
├── tests/
│
└── README.md