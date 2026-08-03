import subprocess
from pathlib import Path
from typing import Any, Protocol


Action = dict[str, Any]
Observation = dict[str, Any]


class Environment(Protocol):
    """Agent 执行动作所依赖的环境接口。"""

    def execute(self, action: Action) -> Observation:
        ...


class LocalEnvironment:
    """在本机工作目录中执行命令。"""

    def __init__(
        self,
        cwd: str | Path | None = None,
        timeout: int = 30,
    ) -> None:
        self.cwd = Path(cwd).resolve() if cwd else Path.cwd()
        self.timeout = timeout

    def execute(self, action: Action) -> Observation:
        command = action.get("command")

        if not isinstance(command, str) or not command.strip():
            raise ValueError("action must contain a non-empty command")

        result = subprocess.run(
            command,
            shell=True,
            cwd=self.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
            check=False,
        )

        return {
            "output": result.stdout,
            "returncode": result.returncode,
            "exception_info": "",
        }