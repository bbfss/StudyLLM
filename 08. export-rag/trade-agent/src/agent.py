from __future__ import annotations

import re
from pathlib import Path

from .memory import CustomerMemory
from .rag import KnowledgeBase
from .schema import AgentState, CustomerProfile


COUNTRY_KEYWORDS = {
    "us": "United States",
    "usa": "United States",
    "america": "United States",
    "uk": "United Kingdom",
    "britain": "United Kingdom",
    "australia": "Australia",
    "canada": "Canada",
    "germany": "Germany",
    "france": "France",
}


class TradeAgent:
    """A small fixed-graph agent for interview-ready demos.

    The workflow mirrors LangGraph concepts:
    - State: AgentState
    - Nodes: methods prefixed with `_node_`
    - Edges: the ordered list in `run`
    - Tools: KnowledgeBase.search and CustomerMemory.save/load
    """

    def __init__(self, data_dir: str | Path, db_path: str | Path = "runtime/customer_memory.db"):
        self.kb = KnowledgeBase(data_dir)
        self.memory = CustomerMemory(db_path)

    def run(self, message: str, customer_id: str = "demo_customer") -> dict:
        state = AgentState(customer_id=customer_id, message=message)
        for node in (
            self._node_classify_intent,
            self._node_retrieve_knowledge,
            self._node_generate_strategy,
            self._node_generate_reply,
            self._node_update_memory,
        ):
            state = node(state)
        return state.to_dict()

    def _node_classify_intent(self, state: AgentState) -> AgentState:
        text = state.message.lower()
        profile = self.memory.load(state.customer_id)
        profile.country = self._extract_country(text) or profile.country
        profile.budget = self._extract_budget(text) or profile.budget

        quantity = self._extract_quantity(text)
        if quantity:
            self._add_interest(profile, f"quantity:{quantity}")

        if any(word in text for word in ["discount", "better price", "cheaper", "lowest", "best price"]):
            state.intent = "price_negotiation"
            state.customer_need = "The customer is asking for a lower price or better deal."
            profile.negotiated_before = True
        elif any(word in text for word in ["shipping", "delivery", "lead time", "arrive", "freight"]):
            state.intent = "shipping"
            state.customer_need = "The customer wants shipping time, delivery method, or freight guidance."
        elif any(word in text for word in ["sample", "try before", "test unit"]):
            state.intent = "sample_request"
            state.customer_need = "The customer wants a sample before bulk order."
        elif any(word in text for word in ["defect", "broken", "complaint", "not working", "refund"]):
            state.intent = "complaint"
            state.customer_need = "The customer reports a quality or after-sales issue."
        elif any(word in text for word in ["logo", "custom", "packaging", "private label"]):
            state.intent = "customization"
            state.customer_need = "The customer asks about logo, packaging, or private label options."
        else:
            state.intent = "general_inquiry"
            state.customer_need = "The customer has a general product or order question."

        state.profile = profile
        state.execution_log.append(f"classify_intent -> {state.intent}")
        return state

    def _node_retrieve_knowledge(self, state: AgentState) -> AgentState:
        query = f"{state.intent} {state.customer_need} {state.message}"
        state.retrieved_policy = self.kb.search(query, top_k=3)
        titles = ", ".join(item.title for item in state.retrieved_policy) or "no evidence"
        state.execution_log.append(f"retrieve_knowledge -> {titles}")
        return state

    def _node_generate_strategy(self, state: AgentState) -> AgentState:
        quantity = self._extract_quantity(state.message.lower())
        if state.intent == "price_negotiation":
            if quantity and quantity >= 500:
                strategy = (
                    "Acknowledge the bulk quantity, avoid an immediate deep discount, "
                    "and offer value-added concession such as free logo printing or sample fee rebate. "
                    "Ask for destination and shipping terms before confirming final price."
                )
            else:
                strategy = (
                    "Ask for target quantity first. For a first-time buyer, keep extra discount within 3% "
                    "and explain MOQ/price tiers."
                )
        elif state.intent == "shipping":
            strategy = "Answer the estimated shipping time, then ask for quantity and destination to calculate freight."
        elif state.intent == "sample_request":
            strategy = "Offer a paid sample, mention sample fee rebate for a qualifying bulk order, and ask for shipping address."
        elif state.intent == "complaint":
            strategy = "Ask for order number, photos, videos, and defective quantity; do not admit liability before verification."
        elif state.intent == "customization":
            strategy = "Explain logo and packaging MOQ, then ask for artwork and expected order quantity."
        else:
            strategy = "Answer briefly using product facts, then ask one concrete question to move the deal forward."

        state.negotiation_strategy = strategy
        state.execution_log.append("generate_strategy -> strategy ready")
        return state

    def _node_generate_reply(self, state: AgentState) -> AgentState:
        evidence_summary = self._evidence_summary(state)
        state.suggested_reply = self._template_reply(state, evidence_summary)
        state.next_action = self._next_action(state)
        state.execution_log.append("generate_reply -> reply ready")
        return state

    def _node_update_memory(self, state: AgentState) -> AgentState:
        profile = state.profile or CustomerProfile(customer_id=state.customer_id)
        if state.intent not in profile.interests:
            profile.interests.append(state.intent)
        profile.next_action = state.next_action
        self.memory.save(profile)
        state.profile = profile
        state.execution_log.append("update_memory -> customer profile saved")
        return state

    def _template_reply(self, state: AgentState, evidence_summary: str) -> str:
        if state.intent == "price_negotiation":
            return (
                "Thanks for sharing your target quantity. For 500 pieces, we can support the bulk order tier and "
                "also discuss value-added support such as free logo printing if the artwork is ready. "
                "Before confirming the final offer, could you please tell me the destination country and preferred shipping method?"
            )
        if state.intent == "shipping":
            country = state.profile.country if state.profile else "your country"
            return (
                f"Shipping to {country} usually depends on quantity and shipping method. Based on our policy, express delivery is "
                "normally faster while sea freight is better for bulk orders. Could you share the order quantity so I can suggest the best option?"
            )
        if state.intent == "sample_request":
            return (
                "Yes, we can arrange a sample for quality checking. The sample is usually prepared within 2-3 working days, "
                "and the sample fee can be refunded once your first bulk order reaches the required quantity. "
                "Please send your delivery address and preferred product color."
            )
        if state.intent == "complaint":
            return (
                "I am sorry to hear that. To check this properly, could you please send the order number, photos or videos, "
                "and the quantity of affected units? Once confirmed, we can arrange replacement support according to our warranty policy."
            )
        if state.intent == "customization":
            return (
                "Yes, we support logo printing and custom packaging. Logo printing starts from 300 pieces, and custom packaging starts from 500 pieces. "
                "Could you share your logo file and expected order quantity?"
            )
        return (
            "Thanks for your message. Our Smart Aroma Diffuser X1 is suitable for wellness shops, gift brands, and home decor sellers. "
            f"Relevant policy found: {evidence_summary}. Could you tell me your target quantity and market?"
        )

    def _next_action(self, state: AgentState) -> str:
        if state.intent == "price_negotiation":
            return "Ask for destination country, shipping terms, and target price before confirming final offer."
        if state.intent == "shipping":
            return "Ask for order quantity and delivery deadline, then estimate freight option."
        if state.intent == "sample_request":
            return "Collect delivery address, color preference, and shipping account if available."
        if state.intent == "complaint":
            return "Collect evidence and order number before offering replacement."
        if state.intent == "customization":
            return "Collect artwork file and expected order quantity."
        return "Ask for target quantity and destination market."

    def _evidence_summary(self, state: AgentState) -> str:
        if not state.retrieved_policy:
            return "No direct policy evidence found."
        return " | ".join(f"{item.source}: {item.title}" for item in state.retrieved_policy)

    def _extract_country(self, text: str) -> str | None:
        for keyword, country in COUNTRY_KEYWORDS.items():
            if keyword in text:
                return country
        return None

    def _extract_budget(self, text: str) -> str | None:
        match = re.search(r"\$\s?([0-9]+(?:\.[0-9]+)?)|usd\s?([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return None
        amount = match.group(1) or match.group(2)
        return f"USD {amount}"

    def _extract_quantity(self, text: str) -> int | None:
        match = re.search(r"(\d{2,6})\s?(pieces|pcs|units|sets|pc)?", text)
        if not match:
            return None
        return int(match.group(1))

    def _add_interest(self, profile: CustomerProfile, value: str) -> None:
        if value not in profile.interests:
            profile.interests.append(value)
