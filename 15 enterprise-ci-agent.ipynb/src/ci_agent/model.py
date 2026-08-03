"""
模型接口模块（Model Interface）

功能说明：
    本模块定义 Agent 与模型之间的交互接口。
    Agent 不关心底层模型具体实现，只要求模型提供统一的 query 方法。

主要组件：
    - Model：
        模型接口协议，规定所有模型后端需要实现的调用方式。

    - DeterministicModel：
        确定性模拟模型，用于在不调用真实大模型、不消耗 API 的情况下，
        稳定复现 Agent Loop、多轮交互以及轨迹（Trajectory）生成过程。

后续扩展方向：
    本模块未来将支持：
        DeterministicModel（测试）
            ↓
        Qwen-Coder / vLLM（真实推理）
            ↓
        rLLM Rollout Worker（轨迹采集）
            ↓
        veRL（GRPO/PPO 强化学习训练）
"""


from copy import deepcopy
from typing import Any, Protocol


Message = dict[str, Any]


class Model(Protocol):
    """Agent 所依赖的最小模型接口。"""

    def query(self, messages: list[Message]) -> Message:
        ...



class DeterministicModel:
    """这里是确定性模型，按照预设顺序返回输出，用于稳定复现 Agent loop。"""

    def __init__(self, outputs: list[Message]) -> None:
        self._outputs = deepcopy(outputs)
        self._index = 0

    def query(self, messages: list[Message]) -> Message:
        if self._index >= len(self._outputs):
            raise RuntimeError("DeterministicModel outputs exhausted")

        output = deepcopy(self._outputs[self._index])
        self._index += 1
        return output