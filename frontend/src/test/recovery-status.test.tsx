import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { api } from '@/api/client';
import { GenerationHistory } from '@/components/GenerationHistory';
import { Layout } from '@/components/Layout';
import { RecoveryStatus } from '@/components/RecoveryStatus';
import { StartupStatus } from '@/components/StartupStatus';
import type { JobSummary, RecoveryCode, RecommendedAction, StartupState } from '@/lib/types';
import { jobFixture } from '@/test/project-fixtures';

vi.mock('@/api/client', async (original) => {
  const actual = await original<typeof import('@/api/client')>();
  return {
    ...actual,
    api: {
      ...actual.api,
      startup: vi.fn(),
      executeOperation: vi.fn(),
      submitLanguage: vi.fn(),
      confirmLanguage: vi.fn(),
    },
  };
});

const recoveryCases: Array<{
  recoveryCode: RecoveryCode;
  recommendedAction: RecommendedAction;
  label: string;
  guidance?: string;
}> = [
  {
    recoveryCode: 'wait',
    recommendedAction: 'wait',
    label: '処理の完了待ち',
    guidance: '処理が完了するまでお待ちください。',
  },
  {
    recoveryCode: 'safe_retry',
    recommendedAction: 'retry_current',
    label: '再実行できます',
    guidance: '現在の設定で新しい生成として再実行できます。',
  },
  {
    recoveryCode: 'external_outcome_unknown',
    recommendedAction: 'check_provider',
    label: '外部処理の結果が不明',
    guidance: '自動で再送しません。外部サービス側の実行履歴を確認してください。',
  },
  {
    recoveryCode: 'refresh_required',
    recommendedAction: 'refresh',
    label: '状態の再取得が必要',
    guidance: '画面を再読み込みして最新の状態を確認してください。',
  },
  {
    recoveryCode: 'cancelled',
    recommendedAction: 'none',
    label: 'キャンセル完了',
  },
  {
    recoveryCode: 'completed',
    recommendedAction: 'none',
    label: '生成完了',
  },
  {
    recoveryCode: 'failed',
    recommendedAction: 'none',
    label: '生成失敗',
  },
];

describe('RecoveryStatus', () => {
  it.each(recoveryCases)(
    'renders exact guidance for $recoveryCode/$recommendedAction without owning an action',
    ({ recoveryCode, recommendedAction, label, guidance }) => {
      const job: JobSummary = jobFixture({
        recovery_code: recoveryCode,
        recommended_action: recommendedAction,
      });

      const { container } = render(<RecoveryStatus job={job} />);

      expect(screen.getAllByRole('status')).toHaveLength(1);
      expect(screen.getByRole('status')).toHaveTextContent(label);
      if (guidance) {
        expect(screen.getByRole('status')).toHaveTextContent(guidance);
      }
      expect(container.querySelector('button')).toBeNull();
    },
  );
});

const startup = (overrides: Partial<StartupState> = {}): StartupState => ({
  status: 'ready',
  reason_code: null,
  message: '起動が完了しました。',
  schema_version: 1,
  backup_available: false,
  ...overrides,
});

describe('GenerationHistory recovery controls', () => {
  it('shows retry only for an authoritative retry-current action', () => {
    render(<GenerationHistory
      jobs={[jobFixture({ recovery_code: 'safe_retry', recommended_action: 'retry_current', retryable: true })]}
      disabled={false}
      running={false}
      onRetry={vi.fn()}
      onCancel={vi.fn()}
    />);

    expect(screen.getByRole('button', { name: '現在の設定で再実行' })).toBeEnabled();
    expect(screen.getAllByRole('status')).toHaveLength(1);
  });

  it('keeps an unknown provider outcome button-free using the typed action', () => {
    render(<GenerationHistory
      jobs={[jobFixture({
        status: 'failed',
        recovery_code: 'external_outcome_unknown',
        recommended_action: 'check_provider',
        retryable: false,
      })]}
      disabled={false}
      running={false}
      onRetry={vi.fn()}
      onCancel={vi.fn()}
    />);

    expect(screen.getByRole('status')).toHaveTextContent('外部処理の結果が不明');
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('renders cancellation wait and completed states without an invalid action', () => {
    const { rerender } = render(<GenerationHistory
      jobs={[jobFixture({
        status: 'running',
        cancel_requested: true,
        recovery_code: 'wait',
        recommended_action: 'wait',
        retryable: false,
      })]}
      disabled={false}
      running
      onRetry={vi.fn()}
      onCancel={vi.fn()}
    />);

    expect(screen.getByRole('button', { name: '停止を待っています' })).toBeDisabled();
    expect(screen.getByRole('status')).toHaveTextContent('処理の完了待ち');

    rerender(<GenerationHistory
      jobs={[jobFixture({
        status: 'failed',
        recovery_code: 'completed',
        recommended_action: 'none',
        retryable: true,
      })]}
      disabled={false}
      running={false}
      onRetry={vi.fn()}
      onCancel={vi.fn()}
    />);

    expect(screen.getByRole('status')).toHaveTextContent('生成完了');
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('follows DOM-order tab navigation and activates retry once with Enter', async () => {
    const user = userEvent.setup();
    const retry = vi.fn();
    render(<>
      <button type="button">履歴の前</button>
      <GenerationHistory
        jobs={[jobFixture({ recovery_code: 'safe_retry', recommended_action: 'retry_current', retryable: true })]}
        disabled={false}
        running={false}
        onRetry={retry}
        onCancel={vi.fn()}
      />
      <button type="button">履歴の後</button>
    </>);

    await user.tab();
    expect(screen.getByRole('button', { name: '履歴の前' })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole('button', { name: '現在の設定で再実行' })).toHaveFocus();
    await user.keyboard('{Enter}');
    expect(retry).toHaveBeenCalledOnce();
    await user.tab();
    expect(screen.getByRole('button', { name: '履歴の後' })).toHaveFocus();
  });

  it('provides structural narrow-layout containment; Task 4 must verify 390px overflow in a browser', () => {
    const recoveryMessage = '外部処理の状態を確認する必要があります。'.repeat(20);
    const { container } = render(<GenerationHistory
      jobs={[jobFixture({
        recovery_message: recoveryMessage,
        recovery_code: 'external_outcome_unknown',
        recommended_action: 'check_provider',
        retryable: false,
      })]}
      disabled={false}
      running={false}
      onRetry={vi.fn()}
      onCancel={vi.fn()}
    />);

    expect(container.firstElementChild).toHaveClass('max-w-full');
    expect(container.querySelector('li')).toHaveClass('min-w-0');
    expect(screen.getByText(recoveryMessage)).toHaveClass('break-words', 'whitespace-pre-wrap');
  });
});

describe('Layout startup integration', () => {
  it('mounts one startup live region above routed content', async () => {
    vi.mocked(api.startup).mockResolvedValue(startup());

    render(<Layout><p>ページ本文</p></Layout>);

    expect(await screen.findByRole('status')).toHaveTextContent('起動が完了しました。');
    expect(screen.getAllByRole('status')).toHaveLength(1);
    expect(screen.getByText('ページ本文')).toBeInTheDocument();
    expect(api.startup).toHaveBeenCalledOnce();
  });
});

describe('StartupStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('polls startup status at a fixed interval until ready and then stops', async () => {
    vi.useFakeTimers();
    vi.mocked(api.startup)
      .mockResolvedValueOnce(startup({
        status: 'starting',
        message: '起動処理中です。',
        schema_version: null,
      }))
      .mockResolvedValueOnce(startup());

    render(<StartupStatus />);

    await act(async () => Promise.resolve());
    expect(screen.getByRole('status')).toHaveTextContent(
      '起動処理中です。起動が完了するまでお待ちください。',
    );
    expect(api.startup).toHaveBeenCalledTimes(1);

    await act(async () => vi.advanceTimersByTimeAsync(2000));

    expect(screen.getByRole('status')).toHaveTextContent('起動が完了しました。');
    expect(api.startup).toHaveBeenCalledTimes(2);

    await act(async () => vi.advanceTimersByTimeAsync(10000));
    expect(api.startup).toHaveBeenCalledTimes(2);
    expect(api.executeOperation).not.toHaveBeenCalled();
    expect(api.submitLanguage).not.toHaveBeenCalled();
    expect(api.confirmLanguage).not.toHaveBeenCalled();
  });

  it('shows the exact ready status', async () => {
    vi.mocked(api.startup).mockResolvedValue(startup());

    render(<StartupStatus />);

    await screen.findByText('起動が完了しました。');
    expect(screen.getByRole('status')).toHaveTextContent('起動が完了しました。');
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows backup restore guidance only when a verified backup is available', async () => {
    vi.mocked(api.startup).mockResolvedValue(startup({
      status: 'migration_failed',
      reason_code: 'migration_failed',
      message: 'データベースの移行に失敗しました。管理者に確認してください。',
      schema_version: null,
      backup_available: true,
    }));

    const { container } = render(<StartupStatus />);

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'データベースの移行に失敗しました。管理者に確認してください。アプリを停止したまま、文書化されたバックアップ復元手順を確認してください。復元後にアプリを再起動してください。',
    );
    expect(container.querySelector('button')).toBeNull();
  });

  it('shows stop, restart, and support guidance without implying a backup exists', async () => {
    vi.mocked(api.startup).mockResolvedValue(startup({
      status: 'migration_failed',
      reason_code: 'backup_failed',
      message: 'データベースを準備できませんでした。',
      schema_version: null,
      backup_available: false,
    }));

    render(<StartupStatus />);

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(
      'データベースを準備できませんでした。アプリを停止して再起動してください。解決しない場合はサポートに連絡してください。',
    );
    expect(alert).not.toHaveTextContent(/バックアップ|復元/);
  });

  it('uses fixed network guidance, hides private details, and explicitly refetches', async () => {
    vi.mocked(api.startup)
      .mockRejectedValueOnce(new Error('C:\\Users\\private-user\\database.sqlite'))
      .mockResolvedValueOnce(startup());

    render(<StartupStatus />);

    expect(await screen.findByRole('alert')).toHaveTextContent(
      '起動状態を確認できません。通信を確認してから再度確認してください。',
    );
    expect(screen.queryByText(/private-user|database\.sqlite/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: '起動状態を再確認' }));

    expect(await screen.findByRole('status')).toHaveTextContent('起動が完了しました。');
    expect(api.startup).toHaveBeenCalledTimes(2);
  });

  it('aborts in-flight polling and clears timers on unmount without stale updates', async () => {
    vi.useFakeTimers();
    let resolvePolling!: (state: StartupState) => void;
    vi.mocked(api.startup)
      .mockResolvedValueOnce(startup({ status: 'starting', message: '起動処理中です。' }))
      .mockReturnValueOnce(new Promise((resolve) => {
        resolvePolling = resolve;
      }));

    const mounted = render(<StartupStatus />);
    await act(async () => Promise.resolve());
    await act(async () => vi.advanceTimersByTimeAsync(2000));

    const calls = (api.startup as unknown as { mock: { calls: Array<[AbortSignal?]> } }).mock.calls;
    const signal = calls[1][0];
    expect(signal).toBeDefined();

    mounted.unmount();
    expect(signal?.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);

    resolvePolling(startup());
    await act(async () => Promise.resolve());
    expect(screen.queryByText('起動が完了しました。')).not.toBeInTheDocument();
  });
});
