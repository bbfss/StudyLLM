from __future__ import annotations

import json
from pathlib import Path

from src.agent import TradeAgent


ROOT = Path(__file__).parent


def main() -> None:
    demo_db = ROOT / "runtime" / "demo_memory.db"
    if demo_db.exists():
        demo_db.unlink()
    agent = TradeAgent(data_dir=ROOT / "data", db_path=demo_db)
    examples = [
        "Can you give me a better price if I order 500 pieces?",
        "How long does shipping to the US take?",
        "Can I get a sample before bulk order?",
    ]
    for idx, message in enumerate(examples, start=1):
        print(f"\n=== Case {idx}: {message} ===")
        result = agent.run(message, customer_id="demo_buyer")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
