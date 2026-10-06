"""Exact cosine (+ optional lexical) ranking of operation versions, never executable requests."""
from __future__ import annotations

import re
import unicodedata

from pydantic import Field

from app.retrieval.contracts import OperationRef, Scalar, SearchScope
from app.retrieval.reader import VerifiedIndex
from app.retrieval.sources import IndexSources
from app.retrieval.vectors import normalize_vector


class RankedCandidate(OperationRef):
    score: Scalar = Field(ge=-1, le=1)
    document_id: str = Field(max_length=200)


# Weight of the lexical (character-bigram) match in a convex mix with the cosine.
# Japanese requests against Japanese annotations match on surface words even when
# the embedding model favours English; the mix stays within [-1, 1] unclamped.
LEXICAL_WEIGHT = 0.35
_IGNORED = re.compile(r"[\s\W_]+")


def _bigrams(text: str) -> set[str]:
    folded = _IGNORED.sub("", unicodedata.normalize("NFKC", text).casefold())
    return {folded[i:i + 2] for i in range(len(folded) - 1)} or ({folded} if folded else set())


def lexical_similarity(query: str, document: str) -> float:
    """Dice coefficient of character bigrams, in [0, 1]."""
    left, right = _bigrams(query), _bigrams(document)
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def rank_operations(index: VerifiedIndex, vector: tuple[float, ...], scope: SearchScope,
                    current_sources: IndexSources, query_text: str | None = None) -> tuple[RankedCandidate, ...]:
    """Maximum document score per exact operation; ties have a stable ID order.

    With ``query_text`` the score is hybrid: (1 - w) * cosine + w * lexical, where
    schema ("input") documents contribute no lexical part.
    """
    query = normalize_vector(vector, index.manifest.profile.dimensions)
    eligible = {d.document_id for d in index.eligible_documents(scope, current_sources)}
    best: dict[tuple[str, int], RankedCandidate] = {}
    for document, other in zip(index.bundle.documents, index.bundle.vectors, strict=True):
        if document.document_id not in eligible:
            continue
        cosine = sum(x * y for x, y in zip(query, other, strict=True))
        if query_text is not None:
            lexical = lexical_similarity(query_text, document.text) if document.kind != "input" else 0.0
            cosine = (1 - LEXICAL_WEIGHT) * cosine + LEXICAL_WEIGHT * lexical
        score = max(-1.0, min(1.0, cosine))
        candidate = RankedCandidate(operation_id=document.operation_id,
            operation_version=document.operation_version, score=score, document_id=document.document_id)
        previous = best.get(document.key)
        if previous is None or (-score, document.document_id) < (-previous.score, previous.document_id):
            best[document.key] = candidate
    return tuple(sorted(best.values(), key=lambda c: (-c.score, c.operation_id, c.operation_version)))
