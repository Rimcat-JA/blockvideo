import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api, ApiError } from '@/api/client';
import { useProjectOperation } from '@/api/useProjectOperation';
import type { OperationResult } from '@/lib/types';

function hook() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderHook(() => useProjectOperation(4), { wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider> });
}
const success: OperationResult = { operation_id: 'project.generation.start', revision: 3, request_id: 'saved', job_id: 9, data: {} };

beforeEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });

describe('durable operation intent', () => {
  it('does not let a late response from an unmounted page erase a newer pending request', async () => {
    let finishOriginal!: (value: OperationResult) => void;
    let failNewer!: (error: Error) => void;
    const execute = vi.spyOn(api, 'executeOperation')
      .mockImplementationOnce(() => new Promise((resolve) => { finishOriginal = resolve; }))
      .mockResolvedValueOnce(success)
      .mockImplementationOnce(() => new Promise((_resolve, reject) => { failNewer = reject; }))
      .mockResolvedValueOnce(success);
    const originalPage = hook();
    act(() => originalPage.result.current.execute('project.generation.start', 3, {}, '生成開始'));
    originalPage.unmount();
    const newPage = hook();
    act(() => newPage.result.current.resend());
    await waitFor(() => expect(newPage.result.current.locked).toBe(false));
    act(() => newPage.result.current.execute('project.settings.update', 3, { subtitle_font_size: 50 }, '設定保存'));
    const newer = execute.mock.calls[2][0];
    await act(async () => finishOriginal(success));
    expect(JSON.parse(sessionStorage.getItem('blockvideo-operation-4') ?? 'null').request).toEqual(newer);
    await act(async () => failNewer(new TypeError('Failed to fetch')));
    newPage.unmount();
    const reloaded = hook();
    expect(reloaded.result.current.pending?.request).toEqual(newer);
    act(() => reloaded.result.current.resend());
    await waitFor(() => expect(execute).toHaveBeenCalledTimes(4));
    expect(execute.mock.calls[3][0]).toEqual(newer);
  });

  it('submits one request even if two clicks arrive before React rerenders', async () => {
    let finish!: (result: OperationResult) => void;
    const execute = vi.spyOn(api, 'executeOperation').mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    const { result } = hook();
    act(() => {
      result.current.execute('project.generation.start', 3, {}, '生成開始');
      result.current.execute('project.generation.start', 3, {}, '生成開始');
    });
    expect(execute).toHaveBeenCalledTimes(1);
    await act(async () => finish(success));
    await waitFor(() => expect(result.current.locked).toBe(false));
  });

  it('uses a new ID for an intentional new execution after success', async () => {
    const execute = vi.spyOn(api, 'executeOperation').mockResolvedValue(success);
    const { result } = hook();
    act(() => result.current.execute('project.generation.start', 3, {}, '生成開始'));
    await waitFor(() => expect(result.current.locked).toBe(false));
    act(() => result.current.execute('project.generation.start', 3, {}, '生成開始'));
    await waitFor(() => expect(execute).toHaveBeenCalledTimes(2));
    expect(execute.mock.calls[0][0].request_id).not.toEqual(execute.mock.calls[1][0].request_id);
  });

  it('keeps the same body and ID after a 503, even if the caller now has a different revision', async () => {
    const execute = vi.spyOn(api, 'executeOperation').mockRejectedValueOnce(new ApiError(503, 'unavailable')).mockResolvedValue(success);
    const { result } = hook();
    act(() => result.current.execute('project.settings.update', 3, { subtitle_font_size: 50 }, '設定保存'));
    await waitFor(() => expect(result.current.isSending).toBe(false));
    expect(result.current.locked).toBe(true);
    act(() => result.current.execute('project.settings.update', 4, { subtitle_font_size: 52 }, '設定保存'));
    expect(execute).toHaveBeenCalledTimes(1);
    act(() => result.current.resend());
    await waitFor(() => expect(execute).toHaveBeenCalledTimes(2));
    expect(execute.mock.calls[1][0]).toEqual(execute.mock.calls[0][0]);
  });

  it('releases a rejected stale request and requires a new intentional action', async () => {
    const execute = vi.spyOn(api, 'executeOperation').mockRejectedValue(new ApiError(409, '設定が変更されました'));
    const { result } = hook();
    act(() => result.current.execute('project.settings.restore', 3, { revision: 1 }, '設定の復元'));
    await waitFor(() => expect(result.current.isSending).toBe(false));
    expect(result.current.pending).toBeNull();
    expect(result.current.error).toContain('設定が変更されました');
    expect(execute).toHaveBeenCalledTimes(1);
  });
});
