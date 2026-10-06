import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ProjectDetailPage } from '@/pages/ProjectDetailPage';
import { ProjectOutputHistory } from '@/components/ProjectOutputHistory';
import { api } from '@/api/client';
import { blockFixture, historyFixture, jobFixture, projectFixture } from '@/test/project-fixtures';

vi.mock('@/api/client', async (original) => {
  const actual = await original<typeof import('@/api/client')>();
  return { ...actual, api: { ...actual.api, startup: vi.fn(), getProject: vi.fn(), listBlocks: vi.fn(), getProjectHistory: vi.fn(), executeOperation: vi.fn() } };
});

function page() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const view = render(<QueryClientProvider client={client}><MemoryRouter initialEntries={['/projects/4']}>
    <Routes><Route path="/projects/:id" element={<ProjectDetailPage />} /></Routes>
  </MemoryRouter></QueryClientProvider>);
  return { ...view, client };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => { resolve = complete; });
  return { promise, resolve };
}

beforeEach(() => {
  vi.clearAllMocks();
  sessionStorage.clear();
  vi.mocked(api.startup).mockResolvedValue({
    status: 'ready', reason_code: null, message: '起動が完了しました。', schema_version: 1, backup_available: false,
  });
  vi.mocked(api.getProject).mockResolvedValue(projectFixture());
  vi.mocked(api.listBlocks).mockResolvedValue([]);
  vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture());
  vi.mocked(api.executeOperation).mockResolvedValue({ operation_id: '', request_id: '', revision: 4, job_id: null, data: {} });
});

describe('output history', () => {
  it.each([
    ['none', '完成動画はまだありません'], ['current', '最新設定の動画'],
    ['stale', '設定は保存済み・動画には未反映'], ['missing', '完成動画のファイルが見つかりません'],
  ] as const)('makes %s state explicit', (output_state, label) => {
    render(<ProjectOutputHistory projectId={4} history={historyFixture({ output_state })} legacyVideo={false} />);
    expect(screen.getByRole('status')).toHaveTextContent(label);
  });

  it('keeps all successful versions accessible and never links missing media', () => {
    const history = historyFixture({ output_state: 'current', current_artifact_id: 20, artifacts: [
      { id: 20, job_id: 8, revision: 3, created_at: '2026-09-19T00:03:00Z', video_url: '/api/video20', subtitle_url: '/api/sub20', is_current: true, available: true },
      { id: 19, job_id: 7, revision: 2, created_at: '2026-09-19T00:02:00Z', video_url: '/api/video19', subtitle_url: null, is_current: false, available: true },
      { id: 18, job_id: 6, revision: 1, created_at: '2026-09-19T00:01:00Z', video_url: '/api/video18', subtitle_url: null, is_current: false, available: false },
    ] });
    const { container } = render(<ProjectOutputHistory projectId={4} history={history} legacyVideo={false} />);
    expect(screen.getByRole('link', { name: '動画 20 を開く・保存' })).toHaveAttribute('href', '/api/video20');
    expect(screen.getByRole('link', { name: '動画 19 を開く・保存' })).toHaveAttribute('href', '/api/video19');
    expect(screen.queryByRole('link', { name: '動画 18 を開く・保存' })).not.toBeInTheDocument();
    expect(screen.getByText('ファイルが見つかりません')).toBeInTheDocument();
    expect(container.querySelector('video')).toHaveAttribute('src', '/api/video20');
  });

  it('offers restoring an earlier available video as the current one', () => {
    const onRestore = vi.fn();
    const history = historyFixture({ output_state: 'current', current_artifact_id: 20, artifacts: [
      { id: 20, job_id: 8, revision: 3, created_at: '2026-09-19T00:03:00Z', video_url: '/api/video20', subtitle_url: null, is_current: true, available: true },
      { id: 19, job_id: 7, revision: 2, created_at: '2026-09-19T00:02:00Z', video_url: '/api/video19', subtitle_url: null, is_current: false, available: true },
      { id: 18, job_id: 6, revision: 1, created_at: '2026-09-19T00:01:00Z', video_url: '/api/video18', subtitle_url: null, is_current: false, available: false },
    ] });
    render(<ProjectOutputHistory projectId={4} history={history} legacyVideo={false} onRestore={onRestore} />);
    expect(screen.queryByRole('button', { name: '動画 20 を現在の動画に戻す' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '動画 18 を現在の動画に戻す' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '動画 19 を現在の動画に戻す' }));
    expect(onRestore).toHaveBeenCalledWith(19);
  });

  it('labels legacy videos whose settings were never recorded', () => {
    render(<ProjectOutputHistory projectId={4} history={historyFixture()} legacyVideo />);
    expect(screen.getByText('設定の対応未確認の以前の動画')).toBeInTheDocument();
  });

  it.each([null, 20])('keeps missing output explicit instead of a legacy preview (pointer %s)', (current_artifact_id) => {
    const { container } = render(<ProjectOutputHistory projectId={4}
      history={historyFixture({ output_state: 'missing', current_artifact_id })} legacyVideo />);
    expect(screen.getByRole('status')).toHaveTextContent('完成動画のファイルが見つかりません');
    expect(screen.queryByText('設定の対応未確認の以前の動画')).not.toBeInTheDocument();
    expect(container.querySelector('video')).toBeNull();
  });
});

describe('project durable controls', () => {
  it.each([
    ['画像だけ再生成', { kind: 'block_visual', block_index: 6 }],
    ['音声だけ再生成', { kind: 'block_audio', block_index: 6 }],
    ['レンダリングのみ再実行', { kind: 'rerender' }],
    ['このブロックから再レンダリング', { kind: 'rerender' }],
  ])('routes %s through the shared durable core with the block index', async (label, args) => {
    vi.mocked(api.listBlocks).mockResolvedValue([blockFixture()]);
    page();
    fireEvent.click(await screen.findByRole('button', { name: label as string }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledWith(expect.objectContaining({
      operation_id: 'project.generation.start', target: { project_id: 4 }, base_revision: 3, arguments: args,
    })));
  });

  it('locks all generation actions around one uncertain block intent and replays it exactly', async () => {
    vi.mocked(api.listBlocks).mockResolvedValue([blockFixture()]);
    vi.mocked(api.executeOperation).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    page();
    fireEvent.click(await screen.findByRole('button', { name: '画像だけ再生成' }));
    const resend = await screen.findByRole('button', { name: '同じ要求を再送する' });
    expect(screen.getByRole('button', { name: '音声だけ再生成' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '現在の設定で新しく生成' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'レンダリングのみ再実行' })).toBeDisabled();
    const original = vi.mocked(api.executeOperation).mock.calls[0][0];
    fireEvent.click(resend);
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.executeOperation).mock.calls[1][0]).toEqual(original);
  });

  it('saves only changed settings, without requesting generation', async () => {
    page();
    fireEvent.change(await screen.findByLabelText('字幕の大きさ（px）'), { target: { value: '50' } });
    fireEvent.click(screen.getByRole('button', { name: '設定を保存' }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledWith(expect.objectContaining({
      operation_id: 'project.settings.update', base_revision: 3, target: { project_id: 4 },
      arguments: { subtitle_font_size: 50 },
    })));
    expect(vi.mocked(api.executeOperation).mock.calls[0][0].generation_requested).toBeUndefined();
  });

  it('restores an arbitrary older saved revision, not just the immediately previous one', async () => {
    page();
    fireEvent.change(await screen.findByLabelText('戻す設定の版'), { target: { value: '1' } });
    fireEvent.click(screen.getByRole('button', { name: '選んだ設定に戻す' }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledWith(expect.objectContaining({
      operation_id: 'project.settings.restore', base_revision: 3, arguments: { revision: 1 },
    })));
  });

  it.each([
    ['字幕の切り替え', 'packed', { subtitle_mode: 'packed' }],
    ['話す速さ', '1.25', { voicevox_speed_scale: 1.25 }],
    ['話者ID', '2', { voicevox_speaker_id: 2 }],
    ['読み上げの間', 'fixed', { narration_pacing_mode: 'fixed' }],
  ])('saves D21 setting %s through the same durable operation without generation', async (label, value, arguments_) => {
    page();
    fireEvent.change(await screen.findByLabelText(label as string), { target: { value } });
    fireEvent.click(screen.getByRole('button', { name: '設定を保存' }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledTimes(1));
    const request = vi.mocked(api.executeOperation).mock.calls[0][0];
    expect(request).toMatchObject({
      operation_id: 'project.settings.update', target: { project_id: 4 }, base_revision: 3,
      arguments: arguments_,
    });
    expect(request.generation_requested).toBeUndefined();
  });

  it('validates and saves a pronunciation edit while preserving existing entries', async () => {
    const existing = { surface: 'GPU', reading: 'ジーピーユー', accent: null };
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({ pronunciation_overrides: [existing] }));
    page();
    fireEvent.click(await screen.findByRole('button', { name: '読み方を追加' }));
    fireEvent.change(screen.getByLabelText('表記 2'), { target: { value: 'API' } });
    fireEvent.change(screen.getByLabelText('読み方（カタカナ）2'), { target: { value: 'エーピーアイ' } });
    fireEvent.change(screen.getByLabelText('アクセント 2'), { target: { value: '99' } });
    expect(screen.getByRole('button', { name: '設定を保存' })).toBeDisabled();
    expect(api.executeOperation).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText('アクセント 2'), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: '設定を保存' }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.executeOperation).mock.calls[0][0]).toMatchObject({
      operation_id: 'project.settings.update', target: { project_id: 4 }, base_revision: 3,
      arguments: { pronunciation_overrides: [existing, { surface: 'API', reading: 'エーピーアイ', accent: null }] },
    });
  });

  it.each(['pending', 'running'])('disables editing during a %s job', async (status) => {
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({
      generation_recovery: { code: 'busy', recommended_action: 'wait' },
    }));
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture({ status })] }));
    page();
    expect(await screen.findByLabelText('字幕の大きさ（px）')).toBeDisabled();
    expect(screen.getByLabelText('戻す設定の版')).toBeDisabled();
    expect(screen.getByRole('button', { name: '現在の設定で新しく生成' })).toBeDisabled();
    expect(screen.getByText(/生成処理が完了するまでお待ちください/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'キャンセルを要求' }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledWith(expect.objectContaining({
      operation_id: 'project.generation.cancel', arguments: { job_id: 8 },
    })));
  });

  it.each([
    ['failed', 'running'],
    ['cancelled', 'pending'],
  ] as const)('hides retry for a %s job while a %s sibling is active', async (terminalStatus, activeStatus) => {
    const blockedTerminal = jobFixture({
      id: 8,
      status: terminalStatus,
      recovery_code: 'wait',
      recommended_action: 'wait',
      retryable: false,
    });
    const activeSibling = jobFixture({ id: 9, status: activeStatus });
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({
      generation_recovery: { code: 'busy', recommended_action: 'wait' },
    }));
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [blockedTerminal, activeSibling] }));

    page();

    await screen.findByText('生成 8');
    expect(screen.queryByRole('button', { name: '現在の設定で再実行' })).not.toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'キャンセルを要求' })).toHaveLength(1);
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it('keeps settings locked while cancellation is requested but not complete', async () => {
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture({ status: 'running', cancel_requested: true })] }));
    page();
    expect(await screen.findByRole('button', { name: '停止を待っています' })).toBeDisabled();
    expect(screen.getByLabelText('字幕の大きさ（px）')).toBeDisabled();
    expect(screen.getByText(/キャンセル要求済み。処理が安全に/)).toBeInTheDocument();
  });

  it('retries as a new request at the current revision while retaining its parent job', async () => {
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture({ input_revision: 1 })] }));
    page();
    fireEvent.click(await screen.findByRole('button', { name: '現在の設定で再実行' }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledWith(expect.objectContaining({
      operation_id: 'project.generation.retry', base_revision: 3, arguments: { job_id: 8 },
    })));
  });

  it.each(['project', 'history'] as const)('disables recovery actions while the %s query refetches', async (query) => {
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture()] }));
    const mounted = page();
    const retry = await screen.findByRole('button', { name: '現在の設定で再実行' });
    expect(retry).toBeEnabled();

    if (query === 'project') {
      const refresh = deferred<ReturnType<typeof projectFixture>>();
      vi.mocked(api.getProject).mockImplementationOnce(() => refresh.promise);
      act(() => { void mounted.client.refetchQueries({ queryKey: ['project', 4] }); });
      await waitFor(() => expect(retry).toBeDisabled());
      refresh.resolve(projectFixture());
    } else {
      const refresh = deferred<ReturnType<typeof historyFixture>>();
      vi.mocked(api.getProjectHistory).mockImplementationOnce(() => refresh.promise);
      act(() => { void mounted.client.refetchQueries({ queryKey: ['history', 4] }); });
      await waitFor(() => expect(retry).toBeDisabled());
      refresh.resolve(historyFixture({ jobs: [jobFixture()] }));
    }

    await waitFor(() => expect(retry).toBeEnabled());
  });

  it('keeps retry locked during revalidation and executes only when refreshed state still authorizes it', async () => {
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture()] }));
    page();
    const retry = await screen.findByRole('button', { name: '現在の設定で再実行' });
    const refreshedProject = deferred<ReturnType<typeof projectFixture>>();
    const refreshedHistory = deferred<ReturnType<typeof historyFixture>>();
    vi.mocked(api.getProject).mockImplementationOnce(() => refreshedProject.promise);
    vi.mocked(api.getProjectHistory).mockImplementationOnce(() => refreshedHistory.promise);

    fireEvent.click(retry);

    await waitFor(() => expect(retry).toBeDisabled());
    expect(api.executeOperation).not.toHaveBeenCalled();
    refreshedProject.resolve(projectFixture());
    refreshedHistory.resolve(historyFixture({ jobs: [jobFixture()] }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledOnce());
  });

  it('refuses retry when the same-revision job changes before execution', async () => {
    vi.mocked(api.getProjectHistory)
      .mockResolvedValueOnce(historyFixture({ jobs: [jobFixture()] }))
      .mockResolvedValue(historyFixture({ jobs: [jobFixture({
        status: 'unknown', recovery_code: 'external_outcome_unknown', recommended_action: 'check_provider', retryable: false,
      })] }));
    page();

    fireEvent.click(await screen.findByRole('button', { name: '現在の設定で再実行' }));

    expect(await screen.findByText(/最新の状態を確認してから、表示された操作を選び直してください/)).toBeInTheDocument();
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it('refuses retry when immediate revalidation finds an active sibling', async () => {
    vi.mocked(api.getProjectHistory)
      .mockResolvedValueOnce(historyFixture({ jobs: [jobFixture()] }))
      .mockResolvedValue(historyFixture({ jobs: [
        jobFixture({ recovery_code: 'wait', recommended_action: 'wait', retryable: false }),
        jobFixture({ id: 9, status: 'running' }),
      ] }));
    page();
    const retry = await screen.findByRole('button', { name: '現在の設定で再実行' });
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({
      generation_recovery: { code: 'busy', recommended_action: 'wait' },
    }));

    fireEvent.click(retry);

    expect(await screen.findByText(/最新の状態を確認してから、表示された操作を選び直してください/)).toBeInTheDocument();
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it('refuses retry when immediate revalidation returns incoherent revisions', async () => {
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture()] }));
    page();
    const retry = await screen.findByRole('button', { name: '現在の設定で再実行' });
    vi.mocked(api.getProject).mockResolvedValueOnce(projectFixture({ revision: 4 }));

    fireEvent.click(retry);

    expect(await screen.findByText(/最新の状態を確認してから、表示された操作を選び直してください/)).toBeInTheDocument();
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it('refuses cancellation when the same-revision job completes before execution', async () => {
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({
      generation_recovery: { code: 'busy', recommended_action: 'wait' },
    }));
    vi.mocked(api.getProjectHistory)
      .mockResolvedValueOnce(historyFixture({ jobs: [jobFixture({ status: 'running' })] }))
      .mockResolvedValue(historyFixture({ jobs: [jobFixture({
        status: 'completed', recovery_code: 'completed', recommended_action: 'none', retryable: false,
      })] }));
    page();

    const cancel = await screen.findByRole('button', { name: 'キャンセルを要求' });
    vi.mocked(api.getProject).mockResolvedValue(projectFixture());
    fireEvent.click(cancel);

    expect(await screen.findByText(/最新の状態を確認してから、表示された操作を選び直してください/)).toBeInTheDocument();
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it('locks stale project/history actions until an explicit refetch supplies matching typed recovery state', async () => {
    const safeRetry = jobFixture({ recovery_code: 'safe_retry', recommended_action: 'retry_current', retryable: true });
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({ revision: 4 }));
    vi.mocked(api.getProjectHistory)
      .mockResolvedValueOnce(historyFixture({ revision: 3, jobs: [safeRetry] }))
      .mockResolvedValue(historyFixture({ revision: 4, jobs: [safeRetry] }));

    page();

    expect(await screen.findByRole('button', { name: '現在の設定で再実行' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '現在の設定で新しく生成' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '最新の状態を再取得' }));

    await waitFor(() => expect(screen.getByRole('button', { name: '現在の設定で再実行' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: '現在の設定で再実行' }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledWith(expect.objectContaining({
      operation_id: 'project.generation.retry', base_revision: 4, arguments: { job_id: 8 },
    })));
  });

  it('locks duplicate retry activation before a second operation can be submitted', async () => {
    let finish!: (value: { operation_id: string; request_id: string; revision: number; job_id: number | null; data: Record<string, unknown> }) => void;
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({
      jobs: [jobFixture({ recovery_code: 'safe_retry', recommended_action: 'retry_current', retryable: true })],
    }));
    vi.mocked(api.executeOperation).mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    page();

    const retry = await screen.findByRole('button', { name: '現在の設定で再実行' });
    const projectCalls = vi.mocked(api.getProject).mock.calls.length;
    const historyCalls = vi.mocked(api.getProjectHistory).mock.calls.length;
    fireEvent.click(retry);
    fireEvent.click(retry);

    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledOnce());
    expect(api.getProject).toHaveBeenCalledTimes(projectCalls + 1);
    expect(api.getProjectHistory).toHaveBeenCalledTimes(historyCalls + 1);
    expect(retry).toBeDisabled();
    finish({ operation_id: 'project.generation.retry', request_id: 'retry', revision: 3, job_id: 9, data: {} });
    await waitFor(() => expect(screen.getByText('現在の設定での再実行を受け付けました。')).toBeInTheDocument());
  });

  it('shows unknown external outcomes and blocks silently resending or starting again', async () => {
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({
      generation_recovery: {
        code: 'external_outcome_unknown',
        recommended_action: 'check_provider',
      },
    }));
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture({
      status: 'failed', recovery_code: 'external_outcome_unknown', recommended_action: 'check_provider',
      retryable: false, recovery_message: '外部送信後にプロセスが終了しました。', retry_blocked_reason: '外部処理の結果を確認してください。',
    })] }));
    page();
    expect(await screen.findByText(/自動で再送しません/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '現在の設定で再実行' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '現在の設定で新しく生成' })).toBeDisabled();
    expect(screen.getByText('外部処理の結果を確認してください。')).toBeInTheDocument();
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it.each([
    ['busy', 'wait', '生成処理が完了するまでお待ちください。停止要求後も、安全に停止するまで生成操作はできません。'],
    ['external_outcome_unknown', 'check_provider', '外部処理の結果が未確定です。自動で再送せず、外部サービス側の実行履歴を確認してください。'],
  ] as const)('uses project-level %s recovery when the blocker is outside history', async (code, action, guidance) => {
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({
      generation_recovery: { code, recommended_action: action },
    }));
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [] }));
    vi.mocked(api.listBlocks).mockResolvedValue([blockFixture()]);

    page();

    expect(await screen.findByText(guidance)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '現在の設定で新しく生成' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'レンダリングのみ再実行' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '画像だけ再生成' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '音声だけ再生成' })).toBeDisabled();
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it('disables an otherwise safe retry when project-level recovery is unknown', async () => {
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({
      generation_recovery: {
        code: 'external_outcome_unknown',
        recommended_action: 'check_provider',
      },
    }));
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture()] }));

    page();

    expect(await screen.findByRole('button', { name: '現在の設定で再実行' })).toBeDisabled();
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it('refuses retry when refreshed project-level recovery no longer permits generation', async () => {
    vi.mocked(api.getProjectHistory).mockResolvedValue(historyFixture({ jobs: [jobFixture()] }));
    page();

    const retry = await screen.findByRole('button', { name: '現在の設定で再実行' });
    vi.mocked(api.getProject).mockResolvedValue(projectFixture({
      generation_recovery: {
        code: 'external_outcome_unknown',
        recommended_action: 'check_provider',
      },
    }));
    fireEvent.click(retry);

    expect(await screen.findByText(/最新の状態を確認してから、表示された操作を選び直してください/)).toBeInTheDocument();
    expect(api.executeOperation).not.toHaveBeenCalled();
  });

  it('locks actions when current job/history state cannot be loaded', async () => {
    vi.mocked(api.getProjectHistory).mockRejectedValue(new Error('offline'));
    page();
    await screen.findByText(/履歴を取得できません/);
    expect(screen.getByLabelText('字幕の大きさ（px）')).toBeDisabled();
    expect(screen.getByRole('button', { name: '現在の設定で新しく生成' })).toBeDisabled();
  });

  it('reuses the exact saved request after an uncertain network result, including a page remount', async () => {
    vi.mocked(api.executeOperation).mockRejectedValueOnce(new TypeError('Failed to fetch'));
    const mounted = page();
    fireEvent.click(await screen.findByRole('button', { name: '現在の設定で新しく生成' }));
    await screen.findByRole('button', { name: '同じ要求を再送する' });
    const original = vi.mocked(api.executeOperation).mock.calls[0][0];
    mounted.unmount();
    page();
    fireEvent.click(await screen.findByRole('button', { name: '同じ要求を再送する' }));
    await waitFor(() => expect(api.executeOperation).toHaveBeenCalledTimes(2));
    expect(vi.mocked(api.executeOperation).mock.calls[1][0]).toEqual(original);
    await waitFor(() => expect(sessionStorage.getItem('blockvideo-operation-4')).toBeNull());
  });
});
