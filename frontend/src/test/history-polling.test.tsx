import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { focusManager, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useProjectHistory } from '@/api/hooks';
import { useProjectOperation } from '@/api/useProjectOperation';
import { historyFixture, jobFixture } from '@/test/project-fixtures';

function hook() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return renderHook(() => ({ history: useProjectHistory(4), operation: useProjectOperation(4) }), {
    wrapper: ({ children }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>,
  });
}

beforeEach(() => { vi.useFakeTimers(); sessionStorage.clear(); });
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); focusManager.setFocused(undefined); });

describe('history polling', () => {
  it.each(['completed', 'failed', 'cancelled', 'unknown'])('does not continuously poll %s history', async (status) => {
    const fetchHistory = vi.spyOn(api, 'getProjectHistory').mockResolvedValue(historyFixture({ jobs: [jobFixture({ status })] }));
    hook();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(fetchHistory).toHaveBeenCalledTimes(1);
  });

  it('refreshes after a new request, polls pending work and stops when the result becomes unknown', async () => {
    const fetchHistory = vi.spyOn(api, 'getProjectHistory')
      .mockResolvedValueOnce(historyFixture())
      .mockResolvedValueOnce(historyFixture({ jobs: [jobFixture({ status: 'pending' })] }))
      .mockResolvedValue(historyFixture({ jobs: [jobFixture({ status: 'unknown' })] }));
    vi.spyOn(api, 'executeOperation').mockResolvedValue({ operation_id: 'project.generation.start', request_id: 'test', job_id: 8, revision: 3, data: {} });
    const { result } = hook();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    act(() => result.current.operation.execute('project.generation.start', 3, {}, '生成開始'));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(fetchHistory).toHaveBeenCalledTimes(2);
    await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
    await act(async () => { await vi.advanceTimersByTimeAsync(1); });
    expect(fetchHistory).toHaveBeenCalledTimes(3);
    expect(result.current.history.data?.jobs[0].status).toBe('unknown');
    await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
    expect(fetchHistory).toHaveBeenCalledTimes(3);
  });

  it('rechecks idle unknown outcomes when the user returns to the window', async () => {
    const fetchHistory = vi.spyOn(api, 'getProjectHistory').mockResolvedValue(historyFixture({ jobs: [jobFixture({ status: 'unknown' })] }));
    hook();
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    act(() => focusManager.setFocused(false));
    act(() => focusManager.setFocused(true));
    await act(async () => { await vi.advanceTimersByTimeAsync(0); });
    expect(fetchHistory).toHaveBeenCalledTimes(2);
  });
});
