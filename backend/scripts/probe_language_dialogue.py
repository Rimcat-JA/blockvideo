"""D19 synthetic dialogue plus D17 wording regression; never executes operations."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from app.interpretation.contracts import CandidateRef, DialogueContextTurn, InterpretationInput, MinimalState
from app.interpretation.local_chat import LocalChatAdapter
from app.interpretation.service import Interpreter, _SYSTEM
from app.operations.catalog import load_catalog
from scripts.probe_language_variants import CASES, matches


def prior(text: str, status: str, proposal: dict[str, Any], question: str | None = None) -> tuple[DialogueContextTurn, ...]:
    return (DialogueContextTurn.model_validate({'text': text, 'status': status, 'proposal': proposal, 'question': question}),)


def operation(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {'kind': 'operation', 'operation_id': f'project.{name}', 'operation_version': 1, 'arguments': arguments}


async def run(model: str, output: Path) -> bool:
    catalog = load_catalog(Path(__file__).resolve().parents[1] / 'app/operations/definitions.json')
    refs = tuple(CandidateRef(operation_id=item.operation_id, operation_version=item.operation_version) for item in catalog.definitions)
    question = prior('字幕を大きくして', 'needs_input', {'kind': 'clarification', 'question': '何pxにしますか？', 'missing_fields': ['arguments']}, '何pxにしますか？')
    saved = prior('字幕を56pxにして', 'completed', operation('subtitle-font-size.set', {'value': 56}))
    generation = prior('今の設定で動画を作って', 'ready', operation('generation.start', {'kind': 'full'}))
    reading = prior('APIの読み方を登録して', 'needs_input', {'kind': 'clarification', 'question': 'APIは何と読みますか？', 'missing_fields': ['arguments']}, 'APIは何と読みますか？')
    cases = [(text, expected, arguments, ()) for text, expected, arguments in CASES] + [
        ('56px', 'set', {'value': 56}, question),
        ('60pxでお願いします', 'set', {'value': 60}, question),
        ('違う、少し小さく', 'adjust', {'delta': -2}, saved),
        ('さっきの指定は違います。2px下げて', 'adjust', {'delta': -2}, saved),
        ('いや、64pxに訂正して', 'set', {'value': 64}, saved),
        ('違う', 'needs_input', {}, saved),
        ('やめて', 'dismissed', {}, generation),
        ('今回は作らないで', 'dismissed', {}, generation),
        ('その依頼は取り消す', 'dismissed', {}, generation),
        ('GPUの読み方を登録したい', 'needs_input', {}, ()),
        ('APIの読み方を登録して', 'needs_input', {}, ()),
        ('エーピーアイ', 'settings', {'pronunciation_overrides': [{'surface': 'API', 'reading': 'エーピーアイ', 'accent': None}]}, reading),
    ]
    records: list[dict[str, Any]] = []
    async with LocalChatAdapter('http://127.0.0.1:1234/v1', model, reasoning_effort='none') as adapter:
        interpreter = Interpreter(catalog, adapter)
        for text, expected, arguments, dialogue in cases:
            actual = (await interpreter.preview(InterpretationInput(text=text, candidates=refs, dialogue=dialogue,
                state=MinimalState(selected_project_id=1, revision=3, subtitle_font_size=56, status='pending')))).model_dump(mode='json')
            passed = actual['status'] == 'dismissed' if expected == 'dismissed' else matches(actual, expected, arguments)
            records.append({'text': text, 'expected': expected, 'arguments': arguments,
                            'context': [turn.model_dump(mode='json') for turn in dialogue], 'actual': actual, 'passed': passed})
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({'model': model, 'synthetic_only': True, 'executes_operations': False,
                'system_prompt_sha256': hashlib.sha256(_SYSTEM.encode()).hexdigest(),
                'passed': sum(record['passed'] for record in records), 'total': len(cases), 'completed': len(records),
                'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f'{len(records):02d} {"PASS" if passed else "FAIL"} {expected}', flush=True)
    return all(record['passed'] for record in records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    return 0 if asyncio.run(run(args.model, args.output)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
