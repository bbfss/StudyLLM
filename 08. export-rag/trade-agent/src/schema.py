from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    source: str
    title: str
    content: str
    score: float


@dataclass
class CustomerProfile:
    customer_id: str
    country: str = "unknown"
    budget: str = "unknown"
    interests: list[str] = field(default_factory=list)
    negotiated_before: bool = False
    next_action: str = "Ask one concrete follow-up question."


@dataclass
class AgentState:
    customer_id: str
    message: str
    intent: str = "general_inquiry"
    customer_need: str = ""
    profile: CustomerProfile | None = None
    retrieved_policy: list[Evidence] = field(default_factory=list)
    negotiation_strategy: str = ""
    suggested_reply: str = ""
    next_action: str = ""
    execution_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "intent": self.intent,
            "customer_need": self.customer_need,
            "customer_profile": self.profile.__dict__ if self.profile else {},
            "retrieved_policy": [item.__dict__ for item in self.retrieved_policy],
            "negotiation_strategy": self.negotiation_strategy,
            "suggested_reply": self.suggested_reply,
            "next_action": self.next_action,
            "execution_log": self.execution_log,
        }
