import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LanguageResultCard } from '@/components/LanguageResultCard';
import { languageFixture } from '@/test/language-fixtures';
import { historyFixture, projectFixture } from '@/test/project-fixtures';

describe('semantic candidate record', () => {
  it('separates similarity from probability, counts all calls and explains fallback', () => {
    const response = languageFixture();
    response.mode = 'semantic';
    response.diagnostics = {
      started_at: 1, candidates: [], interpretation_ms: 1800, execution_ms: 2,
      generation_execution_ms: null, guard_code: null,
      retrieval: { policy: 'semantic-5-8-all-v1', index_sha256: null,
        ranking: [{ operation_id: 'project.subtitle-font-size.set', operation_version: 1, score: 0.84, document_id: 'example' }],
        stages: [{ name: 'all_tools', candidates: [], result: 'proposed', chat_calls: 2, elapsed_ms: 1500, request_bytes: 1000, response_bytes: 100 }],
        expansion_count: 1, all_tools_count: 1, embedding_calls: 1, embedding_ms: 10, chat_calls: 4, elapsed_ms: 1700, reason: 'candidate_insufficient' },
    };
    render(<LanguageResultCard response={response} project={projectFixture()} history={historyFixture()}
      unavailable={false} busy={false} disabled={false} onConfirm={vi.fn()} onEdit={vi.fn()} />);
    expect(screen.getByText('最初の候補では十分に判断できなかったため、確認する操作の範囲を広げました。')).toBeInTheDocument();
    expect(screen.getByText(/正解の確率ではなく/)).toBeInTheDocument();
    expect(screen.getByText('モデル呼出し: 4回')).toBeInTheDocument();
    expect(screen.getByText('project.subtitle-font-size.set v1: 0.8400')).toBeInTheDocument();
    expect(screen.queryByText(/84%/)).not.toBeInTheDocument();
    expect(screen.getByText(/トークン数・料金ではありません/)).toBeInTheDocument();
  });
});
