/** v0.2.4 regressions: form errors, unknown routes and unconfirmed requests. */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ProjectForm } from '@/components/ProjectForm';
import { ProjectDetailPage } from '@/pages/ProjectDetailPage';
import { useProjectOperation } from '@/api/useProjectOperation';
import { api, ApiError } from '@/api/client';

vi.mock('@/api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/client')>();
  return {
    ...actual,
    api: {
      speakers: vi.fn().mockResolvedValue({ url: '', speakers: [] }),
      projectDefaults: vi.fn().mockResolvedValue({}),
      createProject: vi.fn(),
      getProject: vi.fn(),
      executeOperation: vi.fn(),
      operationResult: vi.fn(),
    },
  };
});

function providers(children: React.ReactNode, path = '/') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <MemoryRouter initialEntries={[path]}>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  sessionStorage.clear();
});

describe('detailed form', () => {
  it('shows why an emptied number field blocks creation', async () => {
    render(providers(<ProjectForm />));
    fireEvent.change(screen.getByLabelText('動画タイトル'), { target: { value: 'タイトル' } });
    fireEvent.change(screen.getByLabelText('完成済みの日本語台本'),
      { target: { value: 'これは二十文字を超える十分な長さの台本です。' } });
    fireEvent.change(screen.getByLabelText('フォントサイズ'), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'プロジェクトを作成' }));
    expect(await screen.findByText('数値を入力してください')).toBeInTheDocument();
    expect(api.createProject).not.toHaveBeenCalled();
  });
});

describe('routes', () => {
  it('renders a not-found page for a non-numeric project id without calling the API', () => {
    render(providers(<Routes><Route path="/projects/:id" element={<ProjectDetailPage />} /></Routes>, '/projects/abc'));
    expect(screen.getByText('ページが見つかりません')).toBeInTheDocument();
    expect(api.getProject).not.toHaveBeenCalled();
  });
});

describe('unconfirmed operation', () => {
  it('can be discarded once the server confirms nothing was committed', async () => {
    vi.mocked(api.executeOperation).mockRejectedValue(new ApiError(500, 'boom'));
    vi.mocked(api.operationResult).mockRejectedValue(new ApiError(404, 'missing'));
    const { result } = renderHook(() => useProjectOperation(7), {
      wrapper: ({ children }) => providers(children),
    });
    act(() => result.current.execute('project.generation.start', 1, { kind: 'full' }, '生成開始'));
    await waitFor(() => expect(result.current.locked).toBe(true));
    await waitFor(() => expect(result.current.isSending).toBe(false));
    act(() => result.current.discard());
    await waitFor(() => expect(result.current.locked).toBe(false));
    expect(api.operationResult).toHaveBeenCalledTimes(1);
    expect(sessionStorage.getItem('blockvideo-operation-7')).toBeNull();
  });
});
