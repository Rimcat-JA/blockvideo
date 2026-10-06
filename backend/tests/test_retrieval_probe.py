"""Retrieval metrics must not quietly become final task-success metrics."""
from scripts.probe_retrieval import KS, summarize


def test_recall_and_proposal_denominators_are_separate() -> None:
    trace = {"embedding_calls": 1, "chat_calls": 3, "expansion_count": 1, "all_tools_count": 1,
        "elapsed_ms": 300, "stages": [{"request_bytes": 1000, "response_bytes": 100}]}
    result = summarize([
        {"expected_operation": True, "hits": {str(k): k >= 5 for k in KS}, "proposal_match": False, "trace": trace},
        {"expected_operation": False, "hits": {str(k): None for k in KS}, "proposal_match": True, "trace": trace},
        {"skipped": "precheck"},
    ])
    assert result["recall_denominator"] == 1 and result["recall_hits"]["3"] == 0 and result["recall_hits"]["5"] == 1
    assert result["proposal_denominator"] == 2 and result["proposal_matched"] == 1
    assert result["chat_calls"] == 6 and result["embedding_calls"] == 2
    assert result["skipped_prechecks"] == 1


def test_embedding_only_does_not_claim_proposal_accuracy() -> None:
    result = summarize([{"expected_operation": True, "hits": {str(k): True for k in KS}}])
    assert result["proposal_matched"] is None and result["proposal_denominator"] == result["chat_calls"] == 0
