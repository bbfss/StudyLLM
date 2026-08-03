"""
Code Agent 核心文件
messages                  = State
model.query(messages)     = Policy 产生 Action
environment.execute()     = Environment Transition
observation               = 新状态的一部分
submit / step limit       = Terminal

                 Model
                  |
                  |
             query(messages)
                  |
                  ↓
               Action
                  |
                  ↓
           execute_actions()
                  |
                  ↓
            Environment
                  |
                  ↓
             Observation
                  |
                  ↓
              messages
                  |
                  |
              下一轮
              
agent.py
├── 控制多轮循环
├── 调用 Model
├── 调用 Environment
└── 收集 messages

"""



from ci_agent.environment import Environment
from ci_agent.model import Message, Model

from pathlib import Path
from typing import Any

from ci_agent.trajectory import save_trajectory, serialize_trajectory


class Agent:
    """最小多轮 Code Agent。"""

    def __init__(
        self,
        model: Model,
        environment: Environment,
        step_limit: int = 10,
    ) -> None:
        self.model = model
        self.environment = environment
        self.step_limit = step_limit

        self.messages: list[Message] = []
        self.n_calls = 0

    def run(self, task: str) -> dict:
        self.messages = [
            {
                "role": "system",
                "content": "You are a code agent that can execute commands.",
            },
            {
                "role": "user",
                "content": task,
            },
        ]

        for _ in range(self.step_limit):
            finished = self.step()

            if finished:
                break
        # for else: 如果循环正常结束（没有 break），说明超过了 step_limit
        else:
            self.messages.append(
                {
                    "role": "exit",
                    "content": "Step limit exceeded",
                    "extra": {
                        "exit_status": "LimitsExceeded",
                        "submission": "",
                    },
                }
            )

        return self.messages[-1].get("extra", {})

    def step(self) -> bool:
        model_message = self.query()
        return self.execute_actions(model_message)

    def query(self) -> Message:
        self.n_calls += 1

        message = self.model.query(self.messages)
        self.messages.append(message)

        return message

    def execute_actions(self, message: Message) -> bool:
        # 这里假设模型消息中包含一个 "actions" 字段，里面是一个动作列表，每个动作是一个字典，包含 "type" 和其他参数。content 给人看，extra给机器看
        #         {
        #     "role": "assistant",
        #     "content": "我先查看文件",
        #     "extra": {
        #         "actions": [
        #             {
        #                 "type": "command",
        #                 "command": "ls"
        #             }
        #         ]
        #     }
        # }
        actions = message.get("extra", {}).get("actions", [])

        if not actions:
            raise ValueError("model message must contain at least one action")

        for action in actions:
            
            # 如果动作类型是 "submit"，则表示任务完成，把退出状态加入到消息队列里面
            if action.get("type") == "submit":
                submission = action.get("submission", "")

                self.messages.append(
                    {
                        "role": "exit",
                        "content": submission,
                        "extra": {
                            "exit_status": "Submitted",
                            "submission": submission,
                        },
                    }
                )

                return True
            
            observation = self.environment.execute(action)

            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"<returncode>{observation['returncode']}</returncode>\n"
                        f"<output>\n{observation['output']}</output>"
                    ),
                    "extra": {
                        "observation": observation,
                    },
                }
            )

        return False
    
    def serialize(self) -> dict[str, Any]:
        return serialize_trajectory(
            messages=self.messages,
            n_calls=self.n_calls,
        )

    def save(self, path: str | Path) -> dict[str, Any]:
        trajectory = self.serialize()
        save_trajectory(trajectory, path)
        return trajectory