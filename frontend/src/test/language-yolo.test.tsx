import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, renderHook, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useLanguageRequest } from '@/api/useLanguageRequest';
import { readLanguageSession } from '@/api/language-storage';
import { LanguageYoloReport } from '@/components/LanguageYoloReport';
import { languageFixture } from '@/test/language-fixtures';

function wrapper({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{children}</QueryClientProvider>;
}

beforeEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); localStorage.clear(); });

describe('unattended (YOLO) requests', () => {
  it('sends mode yolo only when asked and keeps it in the recoverable session', async () => {
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => ({
      ...languageFixture('yolo'), request_id: body.request_id, execution_mode: 'yolo',
      yolo_report: { guessing_allowed: true, bypassed_guards: [], auto_confirmed: ['project.generation.start'], unresolved: null },
    }));
    const hook = renderHook(() => useLanguageRequest(4), { wrapper });
    act(() => { hook.result.current.submit('動画を作り直して', 3, true); });
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    expect(submit.mock.calls[0][0].mode).toBe('yolo');
    expect(readLanguageSession(4)?.request.mode).toBe('yolo');
  });

  it('omits the mode for a normal request', async () => {
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => ({
      ...languageFixture('normal'), request_id: body.request_id }));
    const hook = renderHook(() => useLanguageRequest(4), { wrapper });
    act(() => { hook.result.current.submit('字幕を56pxにして', 3); });
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    expect(submit.mock.calls[0][0].mode).toBeUndefined();
  });

  it('rejects a stored session with an unknown mode', () => {
    sessionStorage.setItem('blockvideo-language-4', JSON.stringify({ request: { request_id: 'r1', text: 'x',
      target: { selected_project_id: 4 }, base_revision: 1, mode: 'reckless' }, epoch: 'e', action: 'submit' }));
    expect(readLanguageSession(4)).toBeNull();
  });

  it('reports what was guessed and confirmed on the user\'s behalf', () => {
    render(<LanguageYoloReport response={{ ...languageFixture('yolo'), execution_mode: 'yolo', yolo_report: {
      guessing_allowed: true, bypassed_guards: ['subtitle_value'], unresolved: null,
      auto_confirmed: ['project.subtitle-font-size.set', 'project.generation.start'] } }} />);
    expect(screen.getByText('確認なしで自動実行しました（YOLO）')).toBeTruthy();
    expect(screen.getByText('字幕サイズの値を推測しました')).toBeTruthy();
    expect(screen.getByText(/字幕サイズの設定、動画の生成を開始/)).toBeTruthy();
  });

  it('renders nothing for a normal response', () => {
    const { container } = render(<LanguageYoloReport response={languageFixture('normal')} />);
    expect(container.textContent).toBe('');
  });
});
