import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { api } from '@/api/client';
import { LanguageCandidateReadiness } from '@/components/LanguageCandidateReadiness';
import { LanguageResultCard } from '@/components/LanguageResultCard';
import { languageFixture } from '@/test/language-fixtures';
import { projectFixture } from '@/test/project-fixtures';
import type { CandidateReadinessSnapshot } from '@/lib/language-types';

const recorded: CandidateReadinessSnapshot = { observed_at: 1, candidates: [{
  operation_id: 'project.settings.update', operation_version: 1, phase: 'candidate_preview',
  arguments_checked: false, readiness: 'blocked', reason_code: 'project_busy',
  missing_fields: [], project_id: 1, revision: 1,
}] };

afterEach(() => vi.restoreAllMocks());

describe('provisional candidate state', () => {
  it('keeps a past refusal labelled historical after generation finishes', () => {
    const response = languageFixture();
    response.status = 'blocked'; response.executed = false; response.result = null;
    response.failure = { reason_code: 'project_busy', message: 'busy' };
    response.diagnostics = { started_at: 1, candidates: [], interpretation_ms: 1, execution_ms: null,
      generation_execution_ms: null, guard_code: null,
      retrieval: { policy: 'semantic-5-8-all-v1', index_sha256: null, ranking: [],
        stages: [{ name: 'initial', candidates: [], candidate_state: recorded, result: 'proposed',
          chat_calls: 1, elapsed_ms: 1, request_bytes: 1, response_bytes: 1 }],
        expansion_count: 0, all_tools_count: 0, embedding_calls: 1, embedding_ms: 1, chat_calls: 1,
        elapsed_ms: 2, reason: 'ranked' } };
    render(<LanguageResultCard response={response} project={projectFixture({ status: 'completed' })}
      unavailable={false} busy={false} disabled={false} onConfirm={vi.fn()} onEdit={vi.fn()} />);
    expect(screen.getByText('この依頼は実行しませんでした')).toBeInTheDocument();
    expect(screen.getByRole('alert')).toHaveTextContent('依頼の判断時点の理由');
    expect(screen.queryByRole('button', { name: 'この内容で設定を保存' })).not.toBeInTheDocument();
  });

  it('refreshes the reason without executing or replacing the saved snapshot', async () => {
    const fresh: CandidateReadinessSnapshot = { observed_at: 2, candidates: [{ ...recorded.candidates[0],
      readiness: 'needs_input', reason_code: 'arguments_unchecked' }] };
    const read = vi.spyOn(api, 'getCandidateReadiness').mockResolvedValue(fresh);
    const submit = vi.spyOn(api, 'submitLanguage');
    const execute = vi.spyOn(api, 'confirmLanguage');
    render(<LanguageCandidateReadiness requestId="req" recorded={recorded} />);
    expect(screen.getByText('生成中のため変更できません')).toBeInTheDocument();
    expect(screen.getByText(/入力値を確定する前の暫定情報/)).toBeInTheDocument();
    expect(read).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: '候補の状態を更新' }));
    await screen.findByText('値・参照を確認してから判定します');
    expect(screen.queryByText('生成中のため変更できません')).not.toBeInTheDocument();
    expect(recorded.candidates[0].reason_code).toBe('project_busy');
    expect(submit).not.toHaveBeenCalled(); expect(execute).not.toHaveBeenCalled();
  });

  it('labels a failed refresh as historical and never claims readiness', async () => {
    vi.spyOn(api, 'getCandidateReadiness').mockRejectedValue(new Error('offline'));
    render(<LanguageCandidateReadiness requestId="req" recorded={recorded} />);
    fireEvent.click(screen.getByRole('button', { name: '候補の状態を更新' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('前回の確認時点');
    expect(screen.getByText('生成中のため変更できません')).toBeInTheDocument();
  });

  it('aborts a refresh when the selected request leaves the screen', async () => {
    let signal: AbortSignal | undefined;
    vi.spyOn(api, 'getCandidateReadiness').mockImplementation((_id, value) => {
      signal = value; return new Promise(() => {});
    });
    const view = render(<LanguageCandidateReadiness requestId="req" recorded={recorded} />);
    fireEvent.click(screen.getByRole('button', { name: '候補の状態を更新' }));
    await waitFor(() => expect(signal).toBeDefined());
    view.unmount(); expect(signal!.aborted).toBe(true);
  });
});
