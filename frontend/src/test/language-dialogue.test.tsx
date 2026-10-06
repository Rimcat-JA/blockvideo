import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useLanguageRequest, type LanguageController } from '@/api/useLanguageRequest';
import { readLanguageSession, saveLanguageSession } from '@/api/language-storage';
import { LanguagePanel } from '@/components/LanguagePanel';
import { LanguageResultCard } from '@/components/LanguageResultCard';
import { languageFixture, readyFixture } from '@/test/language-fixtures';
import { historyFixture, projectFixture } from '@/test/project-fixtures';
import type { LanguageResponse } from '@/lib/language-types';

function wrapper({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{children}</QueryClientProvider>;
}
const question = () => languageFixture('parent', { status: 'needs_input', executed: false, result: null,
  dialogue_available: true, clarification: { kind: 'clarification', question: '何pxにしますか？', missing_fields: ['arguments'] } });

beforeEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });

describe('linked dialogue requests', () => {
  it('creates a new linked ID for an answer and stores its exact body for recovery', async () => {
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementationOnce(async (body) => ({ ...question(), request_id: body.request_id }));
    const hook = renderHook(() => useLanguageRequest(4), { wrapper });
    act(() => { hook.result.current.submit('字幕を大きくして', 3); });
    await waitFor(() => expect(hook.result.current.response?.status).toBe('needs_input'));
    const parent = hook.result.current.response!.request_id;
    submit.mockReset().mockRejectedValue(new TypeError('lost'));
    act(() => { hook.result.current.continueRequest('56px', 3, 'answer'); hook.result.current.continueRequest('56px', 3, 'answer'); });
    await waitFor(() => expect(hook.result.current.issue).toBeTruthy());
    expect(submit).toHaveBeenCalledTimes(1);
    const body = submit.mock.calls[0][0];
    expect(body.request_id).not.toBe(parent);
    expect(body.continuation).toEqual({ parent_request_id: parent, relation: 'answer' });
    expect(readLanguageSession(4)?.request).toEqual(body);
    expect(readLanguageSession(4)?.parentPreview?.question).toBe('何pxにしますか？');
    act(() => { hook.result.current.resend(); });
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(2));
    expect(submit.mock.calls[1][0]).toEqual(body);
  });

  it('retains separate pending questions on target switch and reloads each using GET only', async () => {
    const pending = question();
    saveLanguageSession(4, { request: { request_id: 'parent', text: '字幕を大きく', target: { selected_project_id: 4 }, base_revision: 3 }, epoch: 'a', action: 'lookup' });
    const lookup = vi.spyOn(api, 'getLanguageRequest').mockResolvedValue(pending);
    const submit = vi.spyOn(api, 'submitLanguage');
    const first = renderHook(() => useLanguageRequest(4), { wrapper });
    await waitFor(() => expect(first.result.current.response?.status).toBe('needs_input'));
    first.unmount();
    const second = renderHook(() => useLanguageRequest(5), { wrapper });
    expect(second.result.current.session).toBeNull();
    expect(second.result.current.continueRequest('56px', 3, 'answer')).toBe(false);
    second.unmount();
    const returned = renderHook(() => useLanguageRequest(4), { wrapper });
    await waitFor(() => expect(returned.result.current.response?.request_id).toBe('parent'));
    expect(lookup).toHaveBeenCalledTimes(2);
    expect(submit).not.toHaveBeenCalled();
  });

  it.each(['answer', 'correction'] as const)('does not send a stale %s', async (relation) => {
    vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => ({ ...question(), request_id: body.request_id }));
    const { result } = renderHook(() => useLanguageRequest(4), { wrapper });
    act(() => { result.current.submit('字幕を大きく', 3); });
    await waitFor(() => expect(result.current.response?.status).toBe('needs_input'));
    act(() => { expect(result.current.continueRequest('56px', 4, relation)).toBe(false); });
    expect(result.current.issue).toContain('設定が変わりました');
    expect(api.submitLanguage).toHaveBeenCalledTimes(1);
  });

  it('refuses a superseded confirmation even with the same revision', async () => {
    vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => ({ ...readyFixture(body.request_id), superseded_by: 'next' }));
    const execute = vi.spyOn(api, 'confirmLanguage');
    const { result } = renderHook(() => useLanguageRequest(4), { wrapper });
    act(() => { result.current.submit('作り直して', 3); });
    await waitFor(() => expect(result.current.response?.status).toBe('ready'));
    act(() => { result.current.confirm(3); });
    expect(execute).not.toHaveBeenCalled();
  });
});

function controller(response: LanguageResponse): LanguageController {
  return { response, session: { request: { request_id: response.request_id, text: '字幕を大きくして', target: { selected_project_id: 4 }, base_revision: 3 }, epoch: 'epoch', action: 'lookup' },
    busy: false, issue: null, rejected: false, uncertain: false, locked: false, activity: null, activityStartedAt: null,
    submit: vi.fn(() => true), continueRequest: vi.fn(() => true), confirm: vi.fn(), resend: vi.fn(), lookup: vi.fn() };
}

describe('dialogue controls', () => {
  it('answers only the missing field while showing the original request and question', () => {
    const command = controller(question());
    render(<LanguagePanel project={projectFixture()} history={historyFixture()} controller={command} disabled={false} unavailable={false} running={false} />);
    fireEvent.change(screen.getByRole('textbox', { name: '確認への回答' }), { target: { value: '56px' } });
    fireEvent.click(screen.getByRole('button', { name: '回答を送信' }));
    expect(command.continueRequest).toHaveBeenCalledWith('56px', 3, 'answer');
    expect(command.submit).not.toHaveBeenCalled();
  });

  it('has an explicit correction mode after a committed setting', () => {
    const command = controller(languageFixture('saved', { dialogue_available: true }));
    render(<LanguagePanel project={projectFixture({ revision: 4 })} history={historyFixture()} controller={command} disabled={false} unavailable={false} running={false} />);
    fireEvent.click(screen.getByRole('button', { name: 'この依頼を訂正' }));
    fireEvent.change(screen.getByRole('textbox', { name: '訂正内容' }), { target: { value: '違う、少し小さく' } });
    fireEvent.click(screen.getByRole('button', { name: '訂正を送信' }));
    expect(command.continueRequest).toHaveBeenCalledWith('違う、少し小さく', 4, 'correction');
  });

  it('explicit dismissal does not send a generation confirmation', () => {
    const command = controller({ ...readyFixture(), dialogue_available: true });
    render(<LanguagePanel project={projectFixture()} history={historyFixture()} controller={command} disabled={false} unavailable={false} running={false} />);
    fireEvent.change(screen.getByRole('textbox', { name: '訂正内容' }), { target: { value: '60px' } });
    fireEvent.click(screen.getByRole('button', { name: 'この依頼を取り下げる' }));
    expect(command.continueRequest).toHaveBeenCalledWith('この依頼を取り下げる', 3, 'dismiss');
    expect(command.confirm).not.toHaveBeenCalled();
    expect(screen.getByRole('textbox', { name: '訂正内容' })).toHaveValue('');
  });

  it.each(['blocked', 'unsupported', 'error'] as const)('does not offer continuation or correction for %s', (status) => {
    const command = controller(languageFixture('rejected', { status, executed: false, result: null, dialogue_available: true }));
    render(<LanguagePanel project={projectFixture()} history={historyFixture()} controller={command} disabled={false} unavailable={false} running={false} />);
    expect(screen.queryByRole('button', { name: 'この依頼を訂正' })).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'この動画への依頼' })).toBeInTheDocument();
  });

  it('shows superseded confirmation as unusable without hiding its original proposal', () => {
    render(<LanguageResultCard response={{ ...readyFixture(), superseded_by: 'new' }} project={projectFixture()} history={historyFixture()}
      unavailable={false} busy={false} disabled={false} onConfirm={vi.fn()} onEdit={vi.fn()} />);
    expect(screen.getByRole('button', { name: 'この内容で動画生成を開始' })).toBeDisabled();
    expect(screen.getByText('この依頼には後の回答・訂正があります。古い内容では実行できません。')).toBeInTheDocument();
  });
});
