"""
Retrieval / Citation Agent (Step 2 — runs BEFORE domain agents)
Retrieves relevant regulation chunks for each field and claim.
Grounding: all subsequent agents use these retrieved chunks.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.core.logging import get_logger
from app.models.schemas import LabelSubmission
from app.rag.retrieval import RAGRetriever

logger = get_logger(__name__)


class RetrievalAgent(BaseAgent):
    """
    Step 2: Retrieve regulation chunks for each field and claim.
    Purely retrieval-based — no LLM calls.
    """

    def __init__(self, retriever: RAGRetriever | None = None, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever or RAGRetriever()

    def process(
        self,
        submission: LabelSubmission,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        retrieved_chunks: dict[str, list[dict[str, Any]]] = {}

        # --- Retrieve for mandatory labelling fields ---
        mandatory_queries = [
            ("product_name", "name of food labelling requirements"),
            ("ingredients_list", "list of ingredients labelling requirements"),
            ("allergen_declaration", "allergen declaration requirements"),
            ("net_quantity", "net quantity declaration requirements"),
            ("nutritional_info", "nutritional information labelling requirements"),
            ("veg_nonveg_mark", "vegetarian non-vegetarian mark labelling"),
            ("fssai_licence", "FSSAI logo licence number labelling"),
            ("dates", "date marking manufacturing expiry best before labelling"),
            ("manufacturer_details", "manufacturer packer address labelling"),
            ("storage_instructions", "storage instructions labelling"),
            ("additives", "food additives declaration INS number"),
            ("batch_lot", "batch lot number traceability"),
        ]

        if submission.country_of_origin or not submission.manufacturer_details:
            mandatory_queries.append(
                ("country_of_origin", "country of origin imported food labelling")
            )

        for field, query in mandatory_queries:
            results = self.retriever.retrieve(query, top_k=3)
            retrieved_chunks[field] = [r.to_dict() for r in results]

        # --- Retrieve for each claim ---
        for i, claim in enumerate(submission.claims):
            query = f"food label claim {claim} conditions requirements"
            results = self.retriever.retrieve(query, top_k=5)
            retrieved_chunks[f"claim_{i}_{claim}"] = [r.to_dict() for r in results]

        # --- Retrieve for additives ---
        for additive in submission.additives:
            query = f"food additive {additive.name} {additive.ins_number or ''} permitted"
            results = self.retriever.retrieve(query, top_k=3)
            retrieved_chunks[f"additive_{additive.name}"] = [r.to_dict() for r in results]

        # --- Retrieve for allergen-specific checks ---
        if submission.allergen_declaration:
            query = "allergen declaration consistency ingredients list requirements"
            results = self.retriever.retrieve(query, top_k=3)
            retrieved_chunks["allergen_consistency"] = [r.to_dict() for r in results]

        # Special queries for prohibited claims if any claims look suspicious
        for claim in submission.claims:
            claim_lower = claim.lower()
            if any(kw in claim_lower for kw in ["cure", "treat", "prevent", "clinical", "doctor", "immunity"]):
                query = f"prohibited claims health claims {claim} not permitted"
                results = self.retriever.retrieve(query, top_k=5)
                retrieved_chunks[f"prohibited_check_{claim}"] = [r.to_dict() for r in results]

        total_chunks = sum(len(v) for v in retrieved_chunks.values())
        logger.info(
            "retrieval_complete",
            agent=self.name,
            fields_queried=len(retrieved_chunks),
            total_chunks=total_chunks,
        )

        return {
            "findings": [],  # Retrieval agent produces no findings, only context
            "retrieved_chunks": retrieved_chunks,
            "total_chunks_retrieved": total_chunks,
            "confidence": 1.0 if total_chunks > 0 else 0.3,
        }
