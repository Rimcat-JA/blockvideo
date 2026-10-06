import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useLanguageRequest } from '@/api/useLanguageRequest';
import { saveLanguageSession } from '@/api/language-storage';
import { LanguageWaiting } from '@/components/LanguageWaiting';
import { LanguageDiagnostics } from '@/components/LanguageDiagnostics';
import { languageFixture, readyFixture } from '@/test/language-fixtures';
import type { LanguageResponse } from '@/lib/language-types';

function Demo() {
  const command = useLanguageRequest(4);
  return <><button onClick={() => command.submit('字幕を56pxに', 3)}>送信</button>
    <button onClick={() => command.confirm(3)}>生成を確認</button>
    <LanguageWaiting controller={command} /><p>{command.issue}</p></>;
}
function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><Demo /></QueryClientProvider>);
}
beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(new Date('2026-09-20T12:00:00Z')); vi.restoreAllMocks(); sessionStorage.clear(); });
afterEach(() => { cleanup(); vi.useRealTimers(); });

describe('operation waiting', () => {
  it('shows a notice after30 seconds without another submit, then stops at a result', async () => {
    let finish!: (result: LanguageResponse) => void;
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementation(() => new Promise(resolve => { finish = resolve; }));
    mount();
    fireEvent.click(screen.getByText('送信'));
    await act(async () => { await vi.advanceTimersByTimeAsync(29000); });
    expect(screen.queryByText(/通常より時間がかかっています/)).not.toBeInTheDocument();
    expect(screen.getByText(/この待ち時間は動画生成の時間とは別/)).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByText(/通常より時間がかかっています/)).toBeInTheDocument();
    fireEvent.click(screen.getByText('送信'));
    expect(submit).toHaveBeenCalledTimes(1);
    await act(async () => finish(languageFixture(submit.mock.calls[0][0].request_id)));
    expect(screen.queryByLabelText('操作の待ち時間')).not.toBeInTheDocument();
  });

  it('restores interpreting elapsed time by GET without resubmission', async () => {
    const startedAt = Date.now() - 45000;
    saveLanguageSession(4, { request: { request_id: 'pending', text: '字幕を56pxに', target: { selected_project_id: 4 }, base_revision: 3 },
      action: 'submit', epoch: 'saved', startedAt });
    const submit = vi.spyOn(api, 'submitLanguage');
    const get = vi.spyOn(api, 'getLanguageRequest').mockResolvedValue(languageFixture('pending', {
      status: 'interpreting', executed: false, result: null, prepared_request: null,
      diagnostics: { started_at: startedAt / 1000, candidates: [], interpretation_ms: null, execution_ms: null, generation_execution_ms: null, guard_code: null },
    }));
    await act(async () => { mount(); });
    expect(screen.getByText(/45秒経過/)).toBeInTheDocument();
    expect(screen.getByText(/通常より時間がかかっています/)).toBeInTheDocument();
    expect(get).toHaveBeenCalledTimes(1);
    expect(submit).not.toHaveBeenCalled();
  });

  it('does not include time spent reading a generation confirmation in acceptance waiting', async () => {
    vi.spyOn(api, 'submitLanguage').mockImplementation(async body => readyFixture(body.request_id));
    const execute = vi.spyOn(api, 'confirmLanguage').mockImplementation(() => new Promise(() => {}));
    mount();
    await act(async () => { fireEvent.click(screen.getByText('送信')); });
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(screen.queryByLabelText('操作の待ち時間')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('生成を確認'));
    fireEvent.click(screen.getByText('生成を確認'));
    expect(screen.getByText(/開始要求の結果を確認中/)).toBeInTheDocument();
    expect(screen.getByText(/0秒経過/)).toBeInTheDocument();
    expect(screen.queryByText(/通常より時間がかかっています/)).not.toBeInTheDocument();
    expect(execute).toHaveBeenCalledTimes(1);
  });

  it('replaces waiting with unknown-delivery guidance after a connection error', async () => {
    const submit = vi.spyOn(api, 'submitLanguage').mockRejectedValue(new TypeError('lost'));
    mount();
    await act(async () => { fireEvent.click(screen.getByText('送信')); });
    expect(screen.queryByLabelText('操作の待ち時間')).not.toBeInTheDocument();
    expect(screen.getByText(/処理が完了している可能性/)).toBeInTheDocument();
    fireEvent.click(screen.getByText('送信'));
    expect(submit).toHaveBeenCalledTimes(1);
  });

  it('shows legacy timing as unrecorded rather than zero', () => {
    render(<LanguageDiagnostics response={languageFixture()} />);
    expect(screen.getByText(/依頼の判断・検証: 未記録/)).toBeInTheDocument();
    expect(screen.getByText(/確定した値/)).toBeInTheDocument();
  });
});
