"""Derive a single confirmation-bound job request from a committed settings save."""
from __future__ import annotations

from copy import deepcopy

from app.language_operations.contracts import LanguageResponse
from app.operations.contracts import OperationRequest, OperationTarget
from app.operations.policies import load_policies


def generation_request(response: LanguageResponse) -> OperationRequest:
    """Only receipt values supply the target and revision; never model output.

    The follow-up operation itself comes from the operation policies.
    """
    if response.result is None or not response.generate_after_save:
        raise ValueError("a committed settings receipt is required")
    follow = load_policies().follow_up_generation
    return OperationRequest(
        operation_id=follow.operation_id, operation_version=follow.operation_version,
        arguments=deepcopy(follow.arguments),
        target=OperationTarget(project_id=response.result.project_id),
        request_id=f"{response.core_request_id}-generation",
        base_revision=response.result.revision,
    )
