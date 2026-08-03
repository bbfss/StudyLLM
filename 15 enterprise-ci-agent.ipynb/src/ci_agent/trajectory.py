"""
轨迹记录模块（Trajectory Module）

功能说明：
    本模块负责保存 Code Agent 与 Environment 交互产生的完整轨迹。
    轨迹包含：
        - 对话历史（messages）
        - 模型调用次数
        - 任务结束状态（exit_status）
        - 最终提交结果（submission）

在 Agent Loop 中的数据流：

    Task
      ↓
    Agent
      ↓
    Model 生成 Action
      ↓
    Environment 执行
      ↓
    Observation 返回
      ↓
    Trajectory 保存


当前版本：
    enterprise-ci-agent-0.1

当前用途：
    用于复现 mini-swe-agent 的基础 trajectory 保存机制，
    支撑后续 Agent 数据分析、调试和实验复现。

未来扩展方向：
    本模块将作为强化学习数据接口的一部分，逐步支持：

    - SFT 训练数据构造
    - DPO preference pair 构造
    - GRPO/PPO rollout 数据记录
    - reward 与 advantage 保存
    - verifier 测试结果记录
    - token 消耗与执行成本统计

设计目标：
    Agent 负责决策流程；
    Environment 负责环境交互；
    Trajectory 负责记录经验数据。

    三者解耦，方便未来接入：
        mini-swe-agent
            ↓
        rLLM trajectory pipeline
            ↓
        veRL RL training
"""
import json
from pathlib import Path
from typing import Any

from ci_agent.model import Message


def serialize_trajectory(
    messages: list[Message],
    n_calls: int,
) -> dict[str, Any]:
    last_message = messages[-1] if messages else {}
    last_extra = last_message.get("extra", {})

    return {
        "trajectory_format": "enterprise-ci-agent-0.1",
        "info": {
            "model_stats": {
                "api_calls": n_calls,
                "instance_cost": 0.0,
            },
            "exit_status": last_extra.get("exit_status", ""),
            "submission": last_extra.get("submission", ""),
        },
        "messages": messages,
    }


def save_trajectory(
    trajectory: dict[str, Any],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            trajectory,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )