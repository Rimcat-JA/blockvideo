import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { api } from '@/api/client';
import { LanguageConnection } from '@/components/LanguageConnection';

vi.mock('@/api/client', () => ({ api: { languageConnection: vi.fn() } }));
const configured = { status: 'configured' as const, model: 'chosen-local-model',
  base_url: 'http://127.0.0.1:1234/v1', cloud_fallback: false as const, operation_mode: 'all_tools' as const };
beforeEach(() => { vi.clearAllMocks(); vi.mocked(api.languageConnection).mockResolvedValue(configured); });
function renderPanel() {
  return render(<QueryClientProvider client={new QueryClient()}><LanguageConnection /></QueryClientProvider>);
}

it('shows selected model and checks only after an explicit click without claiming inference', async () => {
  renderPanel();
  await screen.findByText('言葉で操作するAI：chosen-local-model');
  expect(api.languageConnection).toHaveBeenCalledTimes(1);
  expect(api.languageConnection).toHaveBeenCalledWith();
  vi.mocked(api.languageConnection).mockResolvedValue({ ...configured, status: 'listed' });
  fireEvent.click(screen.getByText('接続を確認'));
  await screen.findByText('接続先の一覧にモデルを確認しました。依頼を送ると実際の処理が始まります。');
  expect(api.languageConnection).toHaveBeenLastCalledWith(true);
});

it('replaces an earlier success with failure and permits another explicit check', async () => {
  renderPanel();
  await screen.findByText('言葉で操作するAI：chosen-local-model');
  vi.mocked(api.languageConnection).mockResolvedValue({ ...configured, status: 'unreachable' });
  fireEvent.click(screen.getByText('接続を確認'));
  await screen.findByText('接続を確認できません。LM Studioの起動とサーバー設定を確認してください。');
  vi.mocked(api.languageConnection).mockRejectedValue(new Error('offline'));
  fireEvent.click(screen.getByText('接続を確認'));
  await screen.findByText('接続情報を取得できません。もう一度確認してください。');
  await waitFor(() => expect(api.languageConnection).toHaveBeenCalledTimes(3));
});

it('shows host mode without claiming that search has succeeded', async () => {
  vi.mocked(api.languageConnection).mockResolvedValue({ ...configured, operation_mode: 'stateful', status: 'unreachable' });
  renderPanel();
  await screen.findByText('操作の判断方式：状態付き検索（接続先の設定）');
  expect(api.languageConnection).toHaveBeenCalledTimes(1);
  expect(screen.queryByText('検索成功')).not.toBeInTheDocument();
});

it('does not guess a mode when connecting to an older server', async () => {
  vi.mocked(api.languageConnection).mockResolvedValue({ ...configured, operation_mode: undefined });
  renderPanel();
  await screen.findByText('言葉で操作するAI：chosen-local-model');
  expect(screen.getByText('操作の判断方式：未確認（接続先の設定）')).toBeInTheDocument();
});
