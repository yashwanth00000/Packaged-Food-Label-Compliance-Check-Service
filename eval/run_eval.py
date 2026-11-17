"""
Evaluation harness — dual-mode and dual-dataset evaluation.

Datasets:
1. Development Dataset (eval/samples.py): 22 tuning samples used for rule/threshold calibration.
2. Held-out Dataset (eval/heldout/samples.py): 8 un-tuned test samples for unbiased generalization check.

Supports:
- --mode mock (deterministic CI)
- --mode real (uses GEMINI_API_KEY)
- --dataset dev | heldout | all (default: all)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.core.config import PROJECT_ROOT as APP_ROOT
from app.core.config import get_settings
from app.models.schemas import ComplianceReport
from app.orchestrator.pipeline import CompliancePipeline
from app.rag.embeddings import HashEmbedding
from app.rag.ingestion import ingest_regulations
from app.rag.retrieval import RAGRetriever
from app.services.llm_provider import LLMService, MockLLM, reset_llm_service
from eval.heldout.samples import HELDOUT_SAMPLES
from eval.samples import EVAL_SAMPLES


def evaluate_dataset(
    dataset_name: str,
    samples: list[dict[str, Any]],
    mode: str,
    pipeline: CompliancePipeline,
    retriever: RAGRetriever,
) -> dict[str, Any]:
    """Run evaluation over a specific dataset and return performance metrics."""
    results = []
    correct = 0
    total = len(samples)

    total_expected_findings = 0
    total_matched_findings = 0

    print(f"\n{'='*75}")
    print(f"  EVALUATION: {dataset_name.upper()} — Mode: {mode.upper()} ({total} Samples)")
    print(f"{'='*75}\n")

    for idx, sample in enumerate(samples, 1):
        name = sample["name"]
        category = sample.get("category", "General")
        label = sample["label"]
        expected = sample["expected_recommendation"]
        expected_findings = sample.get("expected_findings", [])

        start = time.time()
        report: ComplianceReport = pipeline.run(label)
        elapsed = time.time() - start

        actual = report.recommendation
        is_correct = actual == expected

        # Check additional constraints
        constraints_met = True
        constraint_notes = []

        if "must_not_have" in sample:
            if actual in sample["must_not_have"]:
                constraints_met = False
                constraint_notes.append(f"Got {actual.value} which is in must_not_have")

        if "must_have_issue_types" in sample:
            actual_types = {f.issue_type.value for f in report.findings}
            for required_type in sample["must_have_issue_types"]:
                if required_type not in actual_types:
                    constraints_met = False
                    constraint_notes.append(f"Missing expected issue type: {required_type}")

        # Finding-level recall calculation
        sample_matched = 0
        sample_expected_count = len(expected_findings)
        total_expected_findings += sample_expected_count

        for exp in expected_findings:
            exp_field = exp.get("field")
            exp_type = exp.get("issue_type")

            matched = any(
                (not exp_field or f.field == exp_field) and
                (not exp_type or f.issue_type.value == exp_type)
                for f in report.findings
            )
            if matched:
                sample_matched += 1
                total_matched_findings += 1

        finding_recall_str = f"Findings: {sample_matched}/{sample_expected_count}" if sample_expected_count > 0 else "N/A"

        # Citation validity check
        citations_valid = 0
        total_citations = 0
        for f in report.findings:
            if f.chunk_id and f.evidence_quote:
                total_citations += 1
                if retriever.is_valid_citation(f.chunk_id, f.evidence_quote):
                    citations_valid += 1

        passed = is_correct and constraints_met
        if passed:
            correct += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"

        print(
            f"  {status} #{idx:02d} {name:<42} "
            f"Expected: {expected.value:<20} Got: {actual.value:<20} "
            f"({elapsed:.2f}s) | {finding_recall_str}"
        )

        results.append({
            "sample_index": idx,
            "sample_name": name,
            "category": category,
            "expected": expected.value,
            "actual": actual.value,
            "is_correct": is_correct,
            "constraints_met": constraints_met,
            "passed": passed,
            "elapsed_seconds": round(elapsed, 3),
            "findings_count": len(report.findings),
            "sample_matched_findings": sample_matched,
            "sample_expected_findings": sample_expected_count,
            "citations_valid": citations_valid,
            "total_citations": total_citations,
            "has_errors": report.has_errors,
        })

    accuracy = correct / total if total > 0 else 0.0
    finding_recall = total_matched_findings / total_expected_findings if total_expected_findings > 0 else 1.0

    print(f"\n{'-'*75}")
    print(f"  {dataset_name.upper()} SUMMARY:")
    print(f"  Recommendation Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Finding-Level Recall:    {finding_recall:.1%} ({total_matched_findings}/{total_expected_findings})")
    print(f"{'-'*75}\n")

    return {
        "dataset": dataset_name,
        "mode": mode,
        "total_samples": total,
        "correct_recommendations": correct,
        "recommendation_accuracy": accuracy,
        "finding_level_recall": finding_recall,
        "matched_findings": total_matched_findings,
        "expected_findings": total_expected_findings,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run FSSAI Label Compliance Evaluation")
    parser.add_argument("--mode", choices=["mock", "real", "auto"], default="auto", help="Evaluation mode")
    parser.add_argument("--dataset", choices=["dev", "heldout", "all"], default="all", help="Dataset to evaluate")
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if args.mode == "auto":
        mode = "real" if gemini_key else "mock"
    else:
        mode = args.mode

    print(f"Initializing Compliance Evaluation in {mode.upper()} mode...")

    # Set up retriever with hash embeddings (deterministic)
    tmpdir = tempfile.mkdtemp()
    embedder = HashEmbedding()
    ingest_regulations(
        regulations_dir=APP_ROOT / "data" / "regulations",
        persist_dir=tmpdir,
        embedding_provider=embedder,
    )
    retriever = RAGRetriever(persist_dir=tmpdir, embedding_provider=embedder)

    # Set up LLM
    if mode == "mock":
        os.environ["LLM_PROVIDER"] = "mock"
        os.environ["EMBEDDING_PROVIDER"] = "mock"
        get_settings.cache_clear()
        reset_llm_service()
        llm = LLMService(provider=MockLLM())
    else:
        os.environ["LLM_PROVIDER"] = "gemini"
        reset_llm_service()
        from app.services.llm_provider import get_llm_service
        llm = get_llm_service()

    pipeline = CompliancePipeline(llm_service=llm, retriever=retriever)

    all_metrics: dict[str, Any] = {"mode": mode, "timestamp": time.time(), "datasets": {}}

    if args.dataset in ("dev", "all"):
        dev_metrics = evaluate_dataset("Development (Tuning Data)", EVAL_SAMPLES, mode, pipeline, retriever)
        all_metrics["datasets"]["development"] = dev_metrics

    if args.dataset in ("heldout", "all"):
        heldout_metrics = evaluate_dataset("Held-Out (Un-Tuned Evaluation Data)", HELDOUT_SAMPLES, mode, pipeline, retriever)
        all_metrics["datasets"]["heldout"] = heldout_metrics

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"eval_{mode}_{int(time.time())}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Results saved to: {output_file}")

    # Verify pass criteria
    failed = False
    for ds_name, m in all_metrics["datasets"].items():
        if m["recommendation_accuracy"] < 0.75:
            print(f"FAIL: {ds_name} accuracy {m['recommendation_accuracy']:.1%} below 75%")
            failed = True

    if failed:
        sys.exit(1)
    else:
        print("PASS: Compliance Evaluation passed successfully on all evaluated datasets.")
        sys.exit(0)


if __name__ == "__main__":
    main()
