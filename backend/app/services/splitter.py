"""Script splitter.

Strategy:
    1. Ask the LLM to split ``source_script`` into ordered chunks of
       roughly ``splitter_target_chars`` characters without rewriting the
       script content.
    2. Validate the returned JSON against ``SplitPayload`` and confirm
       the concatenation matches the original script.
    3. Retry up to ``splitter_max_attempts`` times with stricter guidance
       before falling back to a deterministic punctuation-based split.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from pydantic import ValidationError

from app.core.config import Settings
from app.core.logging import log
from app.providers.llm import LLMProvider, LLMRequest, LLMMessage
from app.services.stage_schemas import SplitBlock, SplitPayload


SPLIT_SYSTEM_PROMPT = (
    "あなたは日本語台本の編集者です。"
    "ユーザーの台本を意味のまとまりで区切り、JSONだけを返してください。"
)


@dataclass
class SplitResult:
    """Split output plus fallback diagnostics for one script or segment."""

    blocks: list[SplitBlock]
    used_fallback: bool
    attempts: int
    issues: list[str]


def normalize_for_comparison(text: str) -> str:
    """Whitespace normalization used for join-equality checks.

    Any run of whitespace (including newlines) collapses to a single space;
    surrounding whitespace is stripped.
    """
    return re.sub(r"\s+", " ", text).strip()


def normalize_kept(text: str) -> str:
    """Whitespace form we keep in the DB — collapse runs but keep breaks.

    Fenced blocks are copied byte for byte. A ```slide drawing *is* its
    spacing: collapsing runs turns ``│   table   │`` into ``│ table │``, and
    since the renderer draws exactly what it is given, the box reaches the
    screen with its right-hand wall sitting inside itself.
    """
    def collapse(part: str) -> str:
        return re.sub(r"[ \t　]+", " ", part)

    out: list[str] = []
    cursor = 0
    for fence in _CODE_FENCE_RX.finditer(text):
        out.append(collapse(text[cursor : fence.start()]))
        out.append(fence.group(0))
        cursor = fence.end()
    out.append(collapse(text[cursor:]))
    return "".join(out).strip()


def split_joined_source(blocks: list[SplitBlock]) -> str:
    """Concatenate model block source fields for equality diagnostics."""
    return "".join(b.source_text for b in blocks)


def realign_to_source(
    original: str, blocks: list[SplitBlock]
) -> list[SplitBlock] | None:
    """Rebuild ``source_text`` by slicing *original* at the model's boundaries.

    Models reliably choose sensible split points but do not reliably echo
    Japanese text byte-for-byte — they normalise spacing around ASCII tokens
    (``set-cdr! を`` comes back as ``set-cdr!を``), which fails a strict
    equality check and drops the whole split to the deterministic fallback,
    mid-word cuts and all.

    Treating the response as *boundaries* rather than *content* removes that
    failure mode: we match on whitespace-stripped text and take the actual
    characters from the original, so ``source_text`` is verbatim by
    construction. ``tts_text`` is left as the model wrote it — that one is
    meant to differ, since it carries reading aids.

    Returns ``None`` when the blocks genuinely diverge from the script (text
    added, dropped or reordered), which is a real error the caller must retry.
    """
    keep_idx = [i for i, ch in enumerate(original) if not ch.isspace()]
    stripped = "".join(original[i] for i in keep_idx)

    realigned: list[SplitBlock] = []
    pos = 0
    for block in blocks:
        needle = "".join(ch for ch in block.source_text if not ch.isspace())
        if not needle:
            continue
        if not stripped.startswith(needle, pos):
            return None
        start = keep_idx[pos]
        end = keep_idx[pos + len(needle) - 1] + 1
        text = original[start:end].strip()
        if not text:
            return None
        realigned.append(
            SplitBlock(
                index=len(realigned),
                source_text=text,
                tts_text=block.tts_text,
            )
        )
        pos += len(needle)

    if pos != len(stripped) or not realigned:
        return None
    return realigned


_CODE_FENCE_RX = re.compile(r"```[\s\S]*?```")
_CODE_TOKEN = "〔コード{n}〕"
_CODE_TOKEN_RX = re.compile(r"〔コード(\d+)〕")
# Models often re-type the token without its brackets when writing tts_text
# ("コード2"), which would otherwise be read aloud. Strip both shapes.
_CODE_TOKEN_LOOSE_RX = re.compile(r"〔?\s*コード\s*\d+\s*〕?")


def mask_code_blocks(text: str) -> tuple[str, list[str]]:
    """Replace fenced blocks with short tokens before sending text to an LLM.

    These scripts embed Scheme snippets and box-drawing ASCII art
    (``│ ▼ ┌───┬───┐``). Asking a model to echo that back *inside a JSON
    string* reliably produces malformed JSON — unescaped control characters,
    truncated strings — or silently empties the block. Masking keeps the code
    out of the model's output entirely; it is restored verbatim afterwards.
    """
    blocks: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return _CODE_TOKEN.format(n=len(blocks))

    return _CODE_FENCE_RX.sub(_replace, text), blocks


# Box-drawing / arrow glyphs used to hand-draw diagrams in scripts. They are
# meaningless as speech and, left in, are read out one symbol at a time.
_AA_CHARS = "─━│┃┌┏┐┓└┗┘┛├┣┤┫┬┳┴┻┼╋╭╮╯╰═║╔╗╚╝▲▼◀▶△▽◁▷↑↓←→⇒⇐⇔•·"
_AA_LINE_RX = re.compile(rf"^[\s{re.escape(_AA_CHARS)}()\[\]|/\\+*'\"-]*$")
# Emphasis / inline-code markers only. ``#`` and ``>`` are handled separately
# because they carry meaning outside Markdown (``#f`` is Scheme's false, and a
# stray ``>`` may be an arrow), and blanket-stripping them turned ``#f`` into
# the spoken letter "f".
_MD_INLINE_RX = re.compile(r"[*_`~]+")
_LIST_MARKER_RX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
# Markdown structure. These must also match mid-string: by the time narration
# is sanitised the block text has usually had its newlines collapsed, so
# line-anchored patterns alone let ``---`` through into the audio.
_HEADING_RX = re.compile(r"(?:^|(?<=\s))#{1,6}[ \t]+(?:\d+\s*[．.、]\s*)?")
_HRULE_RX = re.compile(r"(?:^|(?<=\s))(?:-{3,}|={3,}|\*{3,})(?=\s|$)")
_BLOCKQUOTE_RX = re.compile(r"(?:^|(?<=\s))>+[ \t]*")
# Scheme literals that would otherwise be read letter by letter.
_SCHEME_LITERALS = {"#t": "真", "#f": "偽"}
_SCHEME_LITERAL_RX = re.compile(r"#[tf]\b")
# Stage directions: a parenthesised action at the start of the script or right
# after a sentence end, whose content reads as a verb ("…と大きく書く").
# Anchoring to a sentence boundary keeps ordinary asides — "リスト（a b c）" —
# intact, which a blanket paren strip would eat.
_STAGE_DIRECTION_RX = re.compile(
    r"(?:^|(?<=[。．！？!?]))\s*（[^）]{3,}?[くぐすずつぬぶむるうふたてい]）\s*"
)


def sanitize_for_narration(text: str) -> str:
    """Strip everything that should be *shown* rather than *spoken*.

    Code listings, hand-drawn ASCII diagrams, Markdown markup and stage
    directions all belong on the page, not in the audio. Read aloud they
    become gibberish — VOICEVOX will happily pronounce ``**``, ``---`` and
    ``┌───┬───┐`` — and they bloat the subtitle, which is generated from this
    narration text.
    """
    body = text or ""
    body = _CODE_FENCE_RX.sub(" ", body)
    body = _CODE_TOKEN_LOOSE_RX.sub(" ", body)
    body = _STAGE_DIRECTION_RX.sub(" ", body)
    # Before the heading strip, so ``#f`` is never mistaken for a heading.
    body = _SCHEME_LITERAL_RX.sub(lambda m: _SCHEME_LITERALS[m.group(0)], body)
    body = _HEADING_RX.sub("", body)
    body = _HRULE_RX.sub(" ", body)
    body = _BLOCKQUOTE_RX.sub("", body)
    body = _LIST_MARKER_RX.sub("", body)

    kept: list[str] = []
    for line in body.splitlines():
        # Drop lines that are purely diagram glyphs / punctuation.
        if line.strip() and _AA_LINE_RX.match(line):
            continue
        kept.append(line)
    body = "\n".join(kept)

    # Remaining stray diagram glyphs inside otherwise-normal sentences.
    body = body.translate({ord(ch): " " for ch in _AA_CHARS})
    body = _MD_INLINE_RX.sub("", body)
    return re.sub(r"\s+", " ", body).strip()


# A sentence left dangling where a code listing used to be. Scripts written as
# documents lean on the code to complete the sentence — "一つのデータを、
# `キー . 値` というペアで表します" — and once the code is unspoken the
# narration collapses to "一つのデータを、というペアで表します".
_NARRATION_GAP_PATTERNS = [
    # The strongest signal: sanitising a fence leaves whitespace behind, and
    # Japanese prose does not otherwise put a space after a comma. Measured at
    # zero hits across scripts that were written for narration.
    re.compile(r"、\s"),
    re.compile(r"[、。]\s*[はがをにともでへ]\s*[、。]"),        # 、は、
    re.compile(r"[をがはにとで]\s*、\s*(?:という|です|なら|は、)"),  # を、という
    re.compile(r"(?:たとえば|つまり|なら|ただし|そして)\s*、\s*(?:[、。]|$)"),
    re.compile(r"(?:^|[。])\s*[はがをにと]\s*、"),              # 。は、
    re.compile(r"[、]\s*(?:です|ます)[。]"),                    # 、です。
    # The block opens on a clause that needed the removed code as its subject.
    re.compile(r"^\s*(?:という|といった|のように|のような|となります|になります)"),
]


def narration_has_gaps(tts_text: str, source_text: str) -> bool:
    """Return whether removing code left the narration ungrammatical.

    Only blocks that actually contained a fence can have holes, so that is
    required first — it keeps the patterns from firing on prose that merely
    happens to use a lot of commas.
    """
    if "```" not in (source_text or ""):
        return False
    body = tts_text or ""
    return any(p.search(body) for p in _NARRATION_GAP_PATTERNS)


_REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"narration": {"type": "string"}},
    "required": ["narration"],
    "additionalProperties": False,
}


def _build_repair_prompt(source_text: str, narration: str) -> LLMRequest:
    """Build the constrained request used to repair code-induced speech gaps."""
    return LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content=(
                    "あなたは日本語の技術解説動画のナレーション編集者です。"
                    "JSON のみで返します。"
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    "次は動画のナレーション原稿です。元の台本にはコードが含まれていますが、"
                    "コードは画面に表示するため読み上げません。"
                    "その結果、いまのナレーションは文が途切れています。\n\n"
                    "コードを読み上げなくても意味が通るよう、ナレーションを書き直してください。\n"
                    "ルール:\n"
                    "- 元の台本にない事実を足さない。内容を変えない。\n"
                    "- コードを指す必要があるときは日本語で言い換える"
                    "（例: (cons 'a 1) →「cons に a と 1 を渡す式」）。\n"
                    "- 記号・コード・マークダウンは書かない。話し言葉のみ。\n"
                    "- 1文は40〜60字程度で「。」で終える。\n"
                    "- 全体の長さは今のナレーションと同程度から少し長い程度に収める。\n\n"
                    "元の台本（コードを含む）:\n"
                    f"```text\n{source_text[:3000]}\n```\n\n"
                    f"いまのナレーション:\n{narration[:2000]}\n\n"
                    '出力は JSON のみ: {"narration": "..."}'
                ),
            ),
        ],
        temperature=0.3,
        max_tokens=2000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "narration", "schema": _REPAIR_SCHEMA},
        },
    )


async def repair_narration_gaps(
    blocks: list[SplitBlock],
    provider: LLMProvider,
    settings: Settings,
) -> int:
    """Rewrite narration that lost its meaning when the code was removed.

    Returns the number of blocks repaired. Failures are non-fatal: a block
    keeps its original narration, which is no worse than before.
    """
    import asyncio

    targets = [b for b in blocks if narration_has_gaps(b.tts_text, b.source_text)]
    if not targets:
        return 0

    sem = asyncio.Semaphore(max(1, settings.narration_repair_concurrency))

    async def repair(block: SplitBlock) -> tuple[SplitBlock, str | None]:
        async with sem:
            try:
                data = await provider.chat_json(
                    _build_repair_prompt(block.source_text, block.tts_text)
                )
            except Exception as exc:  # noqa: BLE001 - keep the original text
                log.warning(
                    "block={idx} ナレーション修復に失敗: {err}",
                    idx=block.index, err=exc,
                )
                return block, None
            text = sanitize_for_narration(str((data or {}).get("narration") or ""))
            return block, text or None

    results = await asyncio.gather(*(repair(b) for b in targets))

    repaired = 0
    for block, text in results:
        if not text:
            continue
        # Guard against the model summarising the block away or padding it out.
        original = len(block.tts_text)
        if not (original * 0.5 <= len(text) <= max(original * 3, original + 200)):
            log.warning(
                "block={idx} 修復結果の長さが不自然なため破棄 ({a} -> {b} 文字)",
                idx=block.index, a=original, b=len(text),
            )
            continue
        block.tts_text = text
        repaired += 1
    return repaired


def unmask_code_blocks(text: str, blocks: list[str]) -> str:
    """Restore masked fenced blocks. Unknown tokens are dropped."""
    def _replace(match: re.Match[str]) -> str:
        idx = int(match.group(1)) - 1
        return blocks[idx] if 0 <= idx < len(blocks) else ""

    return _CODE_TOKEN_RX.sub(_replace, text)


def _build_split_prompt(
    script: str, settings: Settings, *, attempt: int = 1
) -> LLMRequest:
    """Build the source-preserving JSON request for one split attempt."""
    user_prompt = (
        "次の日本語台本を意味のまとまりで分割してください。\n"
        f"目安: 1ブロックあたり {settings.splitter_min_chars}〜{settings.splitter_max_chars} "
        f"文字程度 (通常 {settings.splitter_target_chars} 文字前後)。\n"
        "各ブロックについて 'index', 'source_text', 'tts_text' を返します。\n"
        "source_text は字幕・画面表示用に元台本の該当部分をそのまま保持してください。\n"
        "tts_text は読み上げ用ですが、英字の識別子 (assoc, cdr, table など) は"
        "そのまま英字で残してください。VOICEVOXが正しく読み上げます。"
        "カタカナに置き換えないでください (「アソック」「クッダー」などにしない)。\n"
        "句点や段落、話題転換を優先してください。コード・数式・箇条書きを途中で分断しないでください。\n"
        "内容を追加・削除・要約しないでください。\n"
        "台本中の 〔コード1〕〔コード2〕… はコードブロックの差し込み位置を表す記号です。\n"
        " - source_text では 〔コード1〕 の形のまま、1文字も変えずに残してください。\n"
        " - tts_text からは完全に削除してください(読み上げないため)。「コード1」等と書き換えないこと。\n"
        "出力は必ずJSONのみ。\n"
        '{"blocks": [{"index": 0, "source_text": "...", "tts_text": "..."}, ...]}\n\n'
        "以下の台本を分割してください:\n"
        "```text\n"
        f"{script}\n"
        "```\n"
    )
    if attempt > 1:
        user_prompt += (
            "\n\n注意: 直前の試行で連結が一致しませんでした。"
            " 余計な空白や記号を追加・削除せず、元台本に含まれる文字列をそのまま並べてください。"
        )
    return LLMRequest(
        messages=[
            LLMMessage(role="system", content=SPLIT_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ],
        temperature=0.0,
        # The response restates the segment (twice: source_text + tts_text),
        # so allow several times its length. Providers that require an
        # explicit max_tokens otherwise truncate mid-JSON.
        max_tokens=min(16000, max(2000, len(script) * 6)),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "split",
                "schema": {
                    "type": "object",
                    "properties": {
                        "blocks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": {"type": "integer"},
                                    "source_text": {"type": "string"},
                                    "tts_text": {"type": "string"},
                                },
                                "required": ["index", "source_text", "tts_text"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["blocks"],
                    "additionalProperties": False,
                },
            },
        },
    )


def _deterministic_split(text: str, settings: Settings) -> list[SplitBlock]:
    """Split text locally at punctuation when LLM splitting is unusable."""
    # Use the same routine as the FakeLLMProvider's split function but applied
    # with the configured sizing parameters.
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.findall(r"[^。？！.!?\n]+[。？！.!?]?", normalized)
    chunks: list[str] = []
    buf = ""
    target = settings.splitter_target_chars
    maxc = settings.splitter_max_chars
    minc = settings.splitter_min_chars
    for part in parts:
        if not part.strip():
            continue
        cand = (buf + part).strip() if buf else part.strip()
        if len(cand) <= maxc:
            buf = cand
            continue
        if buf:
            chunks.append(buf)
            buf = part.strip()
        else:
            chunks.append(part.strip())
            buf = ""
    if buf:
        chunks.append(buf)
    merged: list[str] = []
    for chunk in chunks:
        if (
            merged
            and len(merged[-1]) < minc
            and len(merged[-1]) + len(chunk) <= target
        ):
            merged[-1] = merged[-1] + chunk
        else:
            merged.append(chunk)
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= maxc:
            final.append(chunk)
        else:
            for i in range(0, len(chunk), maxc):
                final.append(chunk[i : i + maxc])
    return [SplitBlock(index=i, source_text=c, tts_text=c) for i, c in enumerate(final)]


def segment_script(script: str, *, max_chars: int) -> list[str]:
    """Cut *script* into LLM-sized segments at sentence boundaries.

    The splitter prompt asks the model to echo the script back, so a single
    call on a long script needs output tokens proportional to the whole
    thing — slow, and prone to truncation. Segmenting keeps every call small
    and lets them run concurrently; boundaries land on sentence ends so no
    segment starts mid-thought.
    """
    text = script.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.findall(r"[^。？！\n]+[。？！]?\n*|\n+", text)
    segments: list[str] = []
    buf = ""
    for sentence in sentences:
        if buf and len(buf) + len(sentence) > max_chars:
            segments.append(buf.strip())
            buf = sentence
        else:
            buf += sentence
    if buf.strip():
        segments.append(buf.strip())

    # A single sentence longer than max_chars still has to go somewhere.
    final: list[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            final.append(seg)
        else:
            final.extend(
                seg[i : i + max_chars] for i in range(0, len(seg), max_chars)
            )
    return [s for s in final if s.strip()]


async def split_segment(
    script: str,
    provider: LLMProvider,
    settings: Settings,
    *,
    code_blocks: list[str] | None = None,
) -> SplitResult:
    """Split one LLM-sized segment.

    ``code_blocks`` is supplied when the caller has already masked fenced
    code across the whole script (so segment boundaries never land inside a
    fence); ``script`` is then the masked text and the list is used to
    restore it. When omitted, this masks and restores on its own.
    """
    normalized_script = normalize_kept(script)
    if not normalized_script:
        raise ValueError("台本が空です")

    # The model only ever sees, and only ever has to echo, the masked form.
    if code_blocks is None:
        masked_script, code_blocks = mask_code_blocks(normalized_script)
    else:
        masked_script = normalized_script

    issues: list[str] = []
    last_error: Exception | None = None

    for attempt in range(1, settings.splitter_max_attempts + 1):
        try:
            request = _build_split_prompt(masked_script, settings, attempt=attempt)
            data = await provider.chat_json(request)
            payload = SplitPayload.model_validate(data)
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = exc
            issues.append(f"attempt {attempt}: JSONスキーマ不一致 ({exc.__class__.__name__})")
            continue

        # Realign against the masked text the model actually saw, then put
        # the real code back.
        blocks = realign_to_source(masked_script, payload.blocks)
        if blocks is None:
            joined = split_joined_source(payload.blocks)
            issues.append(
                f"attempt {attempt}: 連結不一致 "
                f"(len llm={len(joined)}, len orig={len(masked_script)})"
            )
            last_error = ValueError("joined != original")
            continue

        restored = [
            SplitBlock(
                index=b.index,
                source_text=unmask_code_blocks(b.source_text, code_blocks),
                # Narration must not read ASCII art aloud, so code tokens are
                # dropped from tts_text rather than restored. Empty narration
                # is left empty here — the spoken cue is added after merging,
                # otherwise it lands mid-sentence inside a merged block.
                tts_text=sanitize_for_narration(b.tts_text),
            )
            for b in blocks
        ]
        restored = [b for b in restored if b.source_text.strip()]
        if not restored:
            issues.append(f"attempt {attempt}: 復元後に空になりました")
            last_error = ValueError("empty after unmask")
            continue
        return SplitResult(
            blocks=restored,
            used_fallback=False,
            attempts=attempt,
            issues=issues,
        )

    # Deterministic fallback. Always succeeds. Operates on the masked text so
    # code fences stay intact, then restores them.
    blocks = [
        SplitBlock(
            index=b.index,
            source_text=unmask_code_blocks(b.source_text, code_blocks),
            tts_text=sanitize_for_narration(b.tts_text),
        )
        for b in _deterministic_split(masked_script, settings)
    ]
    issues.append(
        f"fallback to deterministic split after {settings.splitter_max_attempts} attempts "
        f"(last error: {last_error})"
    )
    return SplitResult(
        blocks=blocks,
        used_fallback=True,
        attempts=settings.splitter_max_attempts,
        issues=issues,
    )


def chunks_to_block_texts(blocks: Iterable[SplitBlock]) -> list[str]:
    """Extract source text from split blocks for callers needing plain chunks."""
    return [b.source_text for b in blocks]


def merge_small_blocks(
    blocks: list[SplitBlock], *, min_chars: int, max_chars: int
) -> list[SplitBlock]:
    """Fold blocks shorter than *min_chars* into their neighbour.

    Masking code fences leaves the model splitting mostly-short prose runs, so
    it happily emits a block per token — a 4.7k script came back as 123 blocks
    averaging ~32 characters, which renders as a stream of two-second clips.
    Merging restores lecture pacing; a code listing merged into the sentence
    that introduces it is exactly the pairing we want on screen anyway.

    Length is measured on narration (``tts_text``) rather than ``source_text``
    because that is what determines a block's on-screen duration — a block
    holding a large code listing still only takes as long as its narration.
    """
    if not blocks:
        return blocks

    merged: list[SplitBlock] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        prev = merged[-1]
        prev_len = len(prev.tts_text.strip())
        cur_len = len(block.tts_text.strip())
        if (prev_len < min_chars or cur_len < min_chars) and prev_len + cur_len <= max_chars:
            merged[-1] = SplitBlock(
                index=prev.index,
                source_text=f"{prev.source_text}\n{block.source_text}".strip(),
                tts_text=f"{prev.tts_text} {block.tts_text}".strip(),
            )
        else:
            merged.append(block)

    return [
        SplitBlock(index=i, source_text=b.source_text, tts_text=b.tts_text)
        for i, b in enumerate(merged)
    ]


async def split_script(
    script: str,
    provider: LLMProvider,
    settings: Settings,
) -> SplitResult:
    """Split a whole script, segmenting it first when it is long.

    Segments are independent, so they are split concurrently and their block
    lists concatenated. A segment that falls back to the deterministic
    splitter only degrades its own span instead of the entire script.
    """
    import asyncio

    normalized_script = normalize_kept(script)
    if not normalized_script:
        raise ValueError("台本が空です")

    # Mask once, up front: segmenting the masked text keeps fenced code
    # atomic, so a segment boundary can never land inside a code block.
    masked_script, code_blocks = mask_code_blocks(normalized_script)

    segments = segment_script(
        masked_script, max_chars=settings.splitter_segment_chars
    )
    if len(segments) <= 1:
        return await split_segment(
            masked_script, provider, settings, code_blocks=code_blocks
        )

    sem = asyncio.Semaphore(max(1, settings.splitter_concurrency))

    async def run(segment: str) -> SplitResult:
        async with sem:
            return await split_segment(
                segment, provider, settings, code_blocks=code_blocks
            )

    results = await asyncio.gather(*(run(seg) for seg in segments))

    blocks: list[SplitBlock] = []
    issues: list[str] = []
    used_fallback = False
    attempts = 0
    for i, res in enumerate(results):
        used_fallback = used_fallback or res.used_fallback
        attempts = max(attempts, res.attempts)
        issues.extend(f"segment {i}: {msg}" for msg in res.issues)
        for block in res.blocks:
            blocks.append(
                SplitBlock(
                    index=len(blocks),
                    source_text=block.source_text,
                    tts_text=block.tts_text,
                )
            )
    blocks = merge_small_blocks(
        blocks,
        min_chars=settings.splitter_min_chars,
        max_chars=settings.splitter_max_chars,
    )
    # Only now, once merging can no longer strand it mid-sentence, give any
    # still-silent block (a bare code listing) something to say.
    for block in blocks:
        if not block.tts_text.strip():
            block.tts_text = "（コードをご覧ください）"
    return SplitResult(
        blocks=blocks,
        used_fallback=used_fallback,
        attempts=attempts,
        issues=issues,
    )
