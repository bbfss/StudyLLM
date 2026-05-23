from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent import TradeAgent


ROOT = Path(__file__).resolve().parents[1]


class TradeAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = TradeAgent(data_dir=ROOT / "data", db_path=Path(self.tmp.name) / "memory.db")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_price_negotiation(self) -> None:
        result = self.agent.run("Can you give me a better price if I order 500 pieces?", "buyer_a")
        self.assertEqual(result["intent"], "price_negotiation")
        self.assertIn("free logo", result["negotiation_strategy"].lower())
        self.assertTrue(result["retrieved_policy"])
        self.assertIn("destination", result["next_action"].lower())

    def test_shipping_to_us(self) -> None:
        result = self.agent.run("How long does shipping to the US take?", "buyer_b")
        self.assertEqual(result["intent"], "shipping")
        self.assertEqual(result["customer_profile"]["country"], "United States")
        self.assertTrue(any("shipping" in item["source"].lower() for item in result["retrieved_policy"]))

    def test_sample_request(self) -> None:
        result = self.agent.run("Can I get a sample before bulk order?", "buyer_c")
        self.assertEqual(result["intent"], "sample_request")
        self.assertIn("sample", result["suggested_reply"].lower())
        self.assertIn("delivery address", result["next_action"].lower())

    def test_memory_across_turns(self) -> None:
        self.agent.run("How long does shipping to the US take?", "buyer_d")
        result = self.agent.run("Can you give me a better price if I order 500 pieces?", "buyer_d")
        self.assertEqual(result["customer_profile"]["country"], "United States")
        self.assertTrue(result["customer_profile"]["negotiated_before"])


if __name__ == "__main__":
    unittest.main()
