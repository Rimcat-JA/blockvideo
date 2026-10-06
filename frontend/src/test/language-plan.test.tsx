import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { api } from '@/api/client';
import { useLanguageRequest } from '@/api/useLanguageRequest';
import { LanguagePlanCard } from '@/components/LanguagePlanCard';
import { languageFixture } from '@/test/language-fixtures';
import { projectFixture } from '@/test/project-fixtures';
import type { LanguageResponse } from '@/lib/language-types';
import type { OperationRequest } from '@/lib/types';

function wrapper({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{children}</QueryClientProvider>;
}

const step = (operation_id: OperationRequest['operation_id'], args: Record<string, unknown>, index: number): OperationRequest => ({
  request_id: `nl-core-test-s${index}`, operation_id, operation_version: 1, target: { project_id: 4 },
  base_revision: 3, arguments: args });
const steps = [step('project.subtitle-font-size.set', { value: 60 }, 1), step('project.generation.start', { kind: 'full' }, 2)];
const plan = (overrides: Partial<LanguageResponse>): LanguageResponse => languageFixture('plan', {
  status: 'ready', executed: false, result: null, requires_confirmation: true, confirmation_token: 'a'.repeat(64),
  prepared_request: null, plan: steps, plan_results: [], ...overrides });

beforeEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });

describe('multi-step plans', () => {
  it('lists the steps and confirms once, including generation', () => {
    const onConfirm = vi.fn();
    render(<LanguagePlanCard response={plan({})} project={projectFixture()} busy={false} disabled={false}
      unavailable={false} onConfirm={onConfirm} onEdit={vi.fn()} />);
    expect(screen.getByText('2件の手順を順に実行する前に確認')).toBeTruthy();
    expect(screen.getByText('字幕サイズの変更')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'この手順で実行し、動画生成も開始' }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it('shows where a plan stopped', () => {
    render(<LanguagePlanCard response={plan({ status: 'blocked', requires_confirmation: false,
      plan_results: [{ operation_id: 'project.subtitle-font-size.set', request_id: 'nl-core-test-s1', revision: 4,
        job_id: null, data: {}, changed: true }],
      failure: { reason_code: 'plan_step_failed', message: '手順2で停止しました。残りの手順は実行していません。' } })}
      project={projectFixture()} busy={false} disabled={false} unavailable={false} onConfirm={vi.fn()} onEdit={vi.fn()} />);
    expect(screen.getByText('手順2で停止しました')).toBeTruthy();
    expect(screen.getByText('実行済み · 設定の版 4')).toBeTruthy();
    expect(screen.getByText('失敗')).toBeTruthy();
  });

  it('confirms a plan with a generation step as a generation confirmation', async () => {
    const submit = vi.spyOn(api, 'submitLanguage').mockImplementation(async (body) => ({ ...plan({}), request_id: body.request_id }));
    const confirm = vi.spyOn(api, 'confirmLanguage').mockImplementation(async (requestId) => ({
      ...plan({ status: 'completed', requires_confirmation: false }), request_id: requestId }));
    const hook = renderHook(() => useLanguageRequest(4), { wrapper });
    act(() => { hook.result.current.submit('字幕を60pxにしてから動画を作り直して', 3); });
    await waitFor(() => expect(hook.result.current.response?.plan).toBeTruthy());
    expect(submit).toHaveBeenCalledTimes(1);
    act(() => { hook.result.current.confirm(3); });
    await waitFor(() => expect(confirm).toHaveBeenCalledTimes(1));
    expect(confirm.mock.calls[0][1]).toEqual({ confirmation_token: 'a'.repeat(64), confirm_generation: true });
  });
});
