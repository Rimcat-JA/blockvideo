import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api, ApiError } from '@/api/client';
import { useLanguageRequest } from '@/api/useLanguageRequest';
import { readLanguageSession, saveLanguageSession } from '@/api/language-storage';
import { compoundFixture, languageFixture, readyFixture } from '@/test/language-fixtures';
import type { LanguageResponse } from '@/lib/language-types';

function hook() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderHook(() => useLanguageRequest(4), { wrapper: ({ children }) =>
    <QueryClientProvider client={client}>{children}</QueryClientProvider> });
}
const idle = () => Promise.resolve();
beforeEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });
afterEach(() => vi.useRealTimers());

describe('durable language requests', () => {
  it('restores compound confirmation without generating, and confirms using the saved revision once', async () => {
    vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => compoundFixture(body.request_id));
    vi.spyOn(api, 'getLanguageRequest').mockImplementation(async (id) => compoundFixture(id));
    const execute = vi.spyOn(api, 'confirmLanguage').mockImplementation(async (id) => languageFixture(id));
    const first = hook();
    act(() => { first.result.current.submit('字幕を56pxにして作り直して', 3); });
    await waitFor(() => expect(first.result.current.response?.status).toBe('ready'));
    expect(execute).not.toHaveBeenCalled();
    first.unmount();
    const reloaded = hook();
    await waitFor(() => expect(reloaded.result.current.response?.status).toBe('ready'));
    act(() => { reloaded.result.current.confirm(3); });
    expect(execute).not.toHaveBeenCalled();
    act(() => { reloaded.result.current.confirm(4); reloaded.result.current.confirm(4); });
    await waitFor(() => expect(reloaded.result.current.busy).toBe(false));
    expect(execute).toHaveBeenCalledTimes(1);
    expect(execute.mock.calls[0][1].confirm_generation).toBe(true);
  });
  it('sends once for two clicks and gives an intentional new request a new ID', async () => {
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => languageFixture(body.request_id));
    const { result } = hook();
    act(() => { result.current.submit('字幕を56pxに', 3); result.current.submit('字幕を56pxに', 3); });
    expect(submit).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(result.current.locked).toBe(false));
    act(() => { result.current.submit('字幕を56pxに', 4); });
    await waitFor(() => expect(result.current.busy).toBe(false));
    expect(submit).toHaveBeenCalledTimes(2);
    expect(submit.mock.calls[0][0].request_id).not.toEqual(submit.mock.calls[1][0].request_id);
  });

  it('keeps an uncertain request body and revision unchanged on explicit resend', async () => {
    const invalidate = vi.spyOn(QueryClient.prototype, 'invalidateQueries');
    const submit = vi.spyOn(api, 'submitLanguage').mockRejectedValueOnce(new ApiError(503, 'unavailable'))
      .mockImplementation(async (body) => languageFixture(body.request_id));
    const { result } = hook();
    act(() => { result.current.submit('字幕を少し大きく', 3); });
    await waitFor(() => expect(result.current.busy).toBe(false));
    expect(result.current.locked).toBe(true);
    expect(result.current.issue).toContain('処理が完了している可能性');
    expect(result.current.response).toBeNull();
    expect(invalidate.mock.calls.map(([options]) => options?.queryKey)).toEqual([
      ['project', 4], ['history', 4], ['blocks', 4],
    ]);
    act(() => { result.current.submit('字幕を64pxに', 7); result.current.resend(); });
    await waitFor(() => expect(result.current.locked).toBe(false));
    expect(submit).toHaveBeenCalledTimes(2);
    expect(submit.mock.calls[1][0]).toEqual(submit.mock.calls[0][0]);
  });

  it('restores after lost acknowledgement using only GET, without another interpretation', async () => {
    const submit = vi.spyOn(api, 'submitLanguage').mockRejectedValue(new TypeError('Failed to fetch'));
    const lookup = vi.spyOn(api, 'getLanguageRequest').mockImplementation(async (id) => languageFixture(id));
    const first = hook();
    act(() => { first.result.current.submit('字幕56px', 3); });
    await waitFor(() => expect(first.result.current.issue).toBeTruthy());
    const saved = readLanguageSession(4)!;
    first.unmount();
    const reloaded = hook();
    await waitFor(() => expect(reloaded.result.current.response?.status).toBe('completed'));
    expect(lookup.mock.calls[0][0]).toBe(saved.request.request_id);
    expect(submit).toHaveBeenCalledTimes(1);
  });

  it('never confirms generation automatically on submit or reload; double-confirm sends once', async () => {
    vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => readyFixture(body.request_id));
    vi.spyOn(api, 'getLanguageRequest').mockImplementation(async (id) => readyFixture(id));
    let finish!: (value: LanguageResponse) => void;
    const execute = vi.spyOn(api, 'confirmLanguage').mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const first = hook();
    act(() => { first.result.current.submit('作り直して', 3); });
    await waitFor(() => expect(first.result.current.response?.status).toBe('ready'));
    expect(execute).not.toHaveBeenCalled();
    first.unmount();
    const reloaded = hook();
    await waitFor(() => expect(reloaded.result.current.response?.status).toBe('ready'));
    expect(execute).not.toHaveBeenCalled();
    act(() => { reloaded.result.current.confirm(3); reloaded.result.current.confirm(3); });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(execute.mock.calls[0][1]).toEqual({ confirmation_token: 'a'.repeat(64), confirm_generation: true });
    await act(async () => finish(languageFixture(execute.mock.calls[0][0])));
  });

  it('replays the same explicit confirmation after a lost response', async () => {
    vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => readyFixture(body.request_id));
    const execute = vi.spyOn(api, 'confirmLanguage').mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockImplementation(async (id) => languageFixture(id));
    const { result } = hook();
    act(() => { result.current.submit('作り直して', 3); });
    await waitFor(() => expect(result.current.response?.status).toBe('ready'));
    act(() => { result.current.confirm(3); });
    await waitFor(() => expect(result.current.issue).toBeTruthy());
    expect(result.current.locked).toBe(true);
    act(() => { result.current.resend(); });
    await waitFor(() => expect(result.current.locked).toBe(false));
    expect(execute.mock.calls[1].slice(0, 2)).toEqual(execute.mock.calls[0].slice(0, 2));
  });

  it('does not confirm a proposal after the visible revision changed', async () => {
    vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => readyFixture(body.request_id));
    const execute = vi.spyOn(api, 'confirmLanguage');
    const { result } = hook();
    act(() => { result.current.submit('作り直して', 3); });
    await waitFor(() => expect(result.current.response?.status).toBe('ready'));
    act(() => { result.current.confirm(4); });
    expect(execute).not.toHaveBeenCalled();
  });

  it('stops before sending if session storage is unavailable', () => {
    const submit = vi.spyOn(api, 'submitLanguage');
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('quota'); });
    const { result } = hook();
    act(() => { result.current.submit('字幕56px', 3); });
    expect(submit).not.toHaveBeenCalled();
    expect(result.current.issue).toContain('送信はしていません');
    expect(result.current.locked).toBe(false);
  });

  it('rejects a response belonging to another project', async () => {
    vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => languageFixture(body.request_id, { project_id: 5 }));
    const { result } = hook();
    act(() => { result.current.submit('字幕56px', 3); });
    await waitFor(() => expect(result.current.issue).toContain('照合'));
    expect(result.current.response).toBeNull();
    expect(result.current.locked).toBe(true);
  });

  it('uses GET while interpreting and never submits from a polling timer', async () => {
    vi.useFakeTimers();
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => languageFixture(body.request_id,
      { status: 'interpreting', result: null, executed: false }));
    const lookup = vi.spyOn(api, 'getLanguageRequest').mockImplementation(async (id) => languageFixture(id));
    const { result } = hook();
    await act(async () => { result.current.submit('字幕56px', 3); await idle(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    expect(submit).toHaveBeenCalledTimes(1);
    expect(lookup).toHaveBeenCalledTimes(1);
    expect(result.current.response?.status).toBe('completed');
  });

  it('does not let an unmounted response overwrite a newer intent', async () => {
    let finish!: (response: LanguageResponse) => void;
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }))
      .mockRejectedValueOnce(new TypeError('Failed to fetch'));
    vi.spyOn(api, 'getLanguageRequest').mockImplementation(async (id) => languageFixture(id));
    const first = hook();
    act(() => { first.result.current.submit('字幕56px', 3); });
    first.unmount();
    const second = hook();
    await waitFor(() => expect(second.result.current.locked).toBe(false));
    act(() => { second.result.current.submit('字幕64px', 4); });
    await waitFor(() => expect(second.result.current.issue).toBeTruthy());
    const newer = readLanguageSession(4);
    await act(async () => finish(languageFixture(submit.mock.calls[0][0].request_id)));
    expect(readLanguageSession(4)).toEqual(newer);
  });

  it('keeps an unexecuted ready proposal locked until the user explicitly resends', async () => {
    saveLanguageSession(4, { request: { request_id: 'crash-window', text: '字幕56px', target: { selected_project_id: 4 }, base_revision: 3 }, epoch: 'epoch', action: 'submit' });
    vi.spyOn(api, 'getLanguageRequest').mockResolvedValue(languageFixture('crash-window', { status: 'ready', executed: false, result: null }));
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => languageFixture(body.request_id));
    const { result } = hook();
    await waitFor(() => expect(result.current.response?.status).toBe('ready'));
    expect(result.current.locked).toBe(true);
    expect(submit).not.toHaveBeenCalled();
    act(() => { result.current.resend(); });
    await waitFor(() => expect(result.current.locked).toBe(false));
    expect(submit.mock.calls[0][0].request_id).toBe('crash-window');
  });
});
