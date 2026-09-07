import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useState } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { QualitySettingsFields } from '@/components/QualitySettingsFields';
import { ProjectForm } from '@/components/ProjectForm';
import { QuickGeneratePage } from '@/pages/QuickGeneratePage';
import { ProjectDetailPage } from '@/pages/ProjectDetailPage';
import { DEFAULT_QUALITY } from '@/lib/validation';
import { api } from '@/api/client';

vi.mock('@/api/client', () => ({
  api: {
    speakers: vi.fn().mockResolvedValue({ url: '', speakers: [] }),
    createProject: vi.fn().mockResolvedValue({ id: 4 }),
    quickCreate: vi.fn().mockResolvedValue({ project: { id: 4 } }),
    getProject: vi.fn().mockResolvedValue({ id: 4, status: 'completed' }),
    listBlocks: vi.fn().mockResolvedValue([]),
    generateAll: vi.fn().mockResolvedValue({ job: { id: 1 } }),
  },
}));

function wrap(component: React.ReactNode, path = '/') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<MemoryRouter initialEntries={[path]}><QueryClientProvider client={client}>{component}</QueryClientProvider></MemoryRouter>);
}

function addReading() {
  fireEvent.click(screen.getByRole('button', { name: '読み方を追加' }));
  fireEvent.change(screen.getByLabelText('表記 1'), { target: { value: 'API' } });
  fireEvent.change(screen.getByLabelText('読み方（カタカナ）1'), { target: { value: 'エーピーアイ' } });
  fireEvent.change(screen.getByLabelText('アクセント 1'), { target: { value: '0' } });
}

beforeEach(() => vi.clearAllMocks());

describe('quality controls', () => {
  it('allows a newly created detailed project to start generation', async () => {
    vi.mocked(api.getProject).mockResolvedValueOnce({ id: 4, status: 'pending', block_count: 0 } as Awaited<ReturnType<typeof api.getProject>>);
    wrap(<Routes><Route path="/projects/:id" element={<ProjectDetailPage />} /></Routes>, '/projects/4');
    const start = await screen.findByRole('button', { name: '生成開始' });
    expect(start).toBeEnabled();
    fireEvent.click(start);
    await waitFor(() => expect(api.generateAll).toHaveBeenCalledWith(4));
  });

  it('edits readings, reports validation, removes rows and retains mode choices', () => {
    function Harness() {
      const [value, setValue] = useState(DEFAULT_QUALITY);
      return <QualitySettingsFields value={value} onChange={setValue} />;
    }
    render(<Harness />);
    expect(screen.getByLabelText('字幕の切り替え')).toHaveValue('sentence');
    expect(screen.getByLabelText('読み上げの間')).toHaveValue('adaptive');
    addReading();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('読み方（カタカナ）1'), { target: { value: 'ひらがな' } });
    expect(screen.getByRole('alert')).toHaveTextContent('全角カタカナ');
    fireEvent.click(screen.getByRole('button', { name: '読み方 1 を削除' }));
    expect(screen.queryByLabelText('表記 1')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('submits detailed quality and pacing settings without losing provider fields', async () => {
    wrap(<ProjectForm />);
    fireEvent.change(screen.getByLabelText('動画タイトル'), { target: { value: '設定テスト' } });
    fireEvent.change(screen.getByLabelText('完成済みの日本語台本'), { target: { value: 'APIの図を説明します。十分な長さの台本を用意して動作を確認します。' } });
    fireEvent.change(screen.getByLabelText('読み上げの間'), { target: { value: 'fixed' } });
    fireEvent.change(screen.getByLabelText(/文末の息継ぎ（秒）/), { target: { value: '1.2' } });
    fireEvent.change(screen.getByLabelText('1ブロックの最大スライド枚数'), { target: { value: '3' } });
    addReading();
    fireEvent.click(screen.getByRole('button', { name: 'プロジェクトを作成' }));
    await waitFor(() => expect(api.createProject).toHaveBeenCalledWith(expect.objectContaining({
      visual_focus_enabled: true, subtitle_mode: 'sentence', narration_pacing_mode: 'fixed',
      narration_sentence_pause_seconds: 1.2, max_slides_per_block: 3,
      pronunciation_overrides: [{ surface: 'API', reading: 'エーピーアイ', accent: 0 }],
      providers: expect.objectContaining({ image_base_url: 'https://api.openai.com/v1' }),
    })));
  });

  it('blocks incomplete quick readings and sends corrected settings to the API', async () => {
    wrap(<QuickGeneratePage />);
    fireEvent.change(screen.getByPlaceholderText('ここに日本語の台本を貼り付けてください…'), { target: { value: 'APIを説明します。' } });
    fireEvent.click(screen.getByText('カスタム設定'));
    fireEvent.click(screen.getByRole('button', { name: '読み方を追加' }));
    const submit = screen.getByRole('button', { name: /クイック生成/ });
    expect(submit).toBeDisabled();
    fireEvent.change(screen.getByLabelText('表記 1'), { target: { value: 'API' } });
    fireEvent.change(screen.getByLabelText('読み方（カタカナ）1'), { target: { value: 'エーピーアイ' } });
    fireEvent.change(screen.getByLabelText('字幕の切り替え'), { target: { value: 'packed' } });
    expect(submit).toBeEnabled();
    fireEvent.click(submit);
    await waitFor(() => expect(api.quickCreate).toHaveBeenCalledWith(expect.objectContaining({
      visual_focus_enabled: true, subtitle_mode: 'packed', narration_pacing_mode: 'adaptive',
      pronunciation_overrides: [{ surface: 'API', reading: 'エーピーアイ', accent: null }],
    })));
  });
});
