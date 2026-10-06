import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { ProjectsPage } from '@/pages/ProjectsPage';
import { api, ApiError } from '@/api/client';
import { projectFixture } from '@/test/project-fixtures';

function page() {
  vi.spyOn(api, 'startup').mockResolvedValue({
    status: 'ready', reason_code: null, message: '起動が完了しました。', schema_version: 1, backup_available: false,
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><MemoryRouter><ProjectsPage /></MemoryRouter></QueryClientProvider>);
}

afterEach(() => vi.restoreAllMocks());

describe('project deletion', () => {
  it('displays a busy rejection instead of silently leaving the project in the list', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue([projectFixture()]);
    vi.spyOn(api, 'deleteProject').mockRejectedValue(new ApiError(409, '生成中は削除できません'));
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    page();
    fireEvent.click(await screen.findByRole('button', { name: '削除' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('削除できませんでした: 生成中は削除できません');
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('保存した動画と設定の履歴も削除'));
    expect(screen.getByRole('link', { name: '説明動画' })).toBeInTheDocument();
  });

  it('disables deletion when generation is already known to be active', async () => {
    vi.spyOn(api, 'listProjects').mockResolvedValue([projectFixture({ status: 'generating' })]);
    const remove = vi.spyOn(api, 'deleteProject');
    page();
    expect(await screen.findByRole('button', { name: '削除' })).toBeDisabled();
    expect(remove).not.toHaveBeenCalled();
  });
});
