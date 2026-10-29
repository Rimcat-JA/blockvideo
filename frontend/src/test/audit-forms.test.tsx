/** Form regressions: off-step stored values, server defaults and the speaker select. */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ProjectSettingsEditor } from '@/components/ProjectSettingsEditor';
import { QuickGeneratePage } from '@/pages/QuickGeneratePage';
import { api } from '@/api/client';
import { projectFixture } from '@/test/project-fixtures';

vi.mock('@/api/client', () => ({
  api: {
    speakers: vi.fn().mockResolvedValue({ url: '', speakers: [] }),
    projectDefaults: vi.fn(),
    quickCreate: vi.fn().mockResolvedValue({ project: { id: 4 } }),
    getProject: vi.fn().mockResolvedValue({ id: 4, status: 'pending' }),
  },
}));

function wrap(component: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<MemoryRouter><QueryClientProvider client={client}>{component}</QueryClientProvider></MemoryRouter>);
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(api.projectDefaults).mockResolvedValue({
    voicevox_speaker_id: 3, voicevox_speed_scale: 1.1, voicevox_pitch_scale: 0,
    voicevox_intonation_scale: 1, voicevox_volume_scale: 1, narration_sentence_pause_seconds: 0.8,
  });
});

describe('settings editor', () => {
  it('saves while another stored value is off the input step', () => {
    const onSave = vi.fn();
    render(<ProjectSettingsEditor project={projectFixture({ voicevox_intonation_scale: 1.25 })}
      disabled={false} onSave={onSave} />);
    fireEvent.change(screen.getByLabelText('話す速さ'), { target: { value: '1.2' } });
    fireEvent.click(screen.getByRole('button', { name: '設定を保存' }));
    expect(onSave).toHaveBeenCalledWith({ voicevox_speed_scale: 1.2 });
  });
});

describe('quick generate defaults', () => {
  it('shows the server defaults and sends a chosen value equal to the built-in default', async () => {
    wrap(<QuickGeneratePage />);
    const speed = await screen.findByLabelText('読み上げ速度');
    await waitFor(() => expect(speed).toHaveValue('1.1'));
    fireEvent.change(speed, { target: { value: '1' } });
    fireEvent.change(screen.getByPlaceholderText('ここに日本語の台本を貼り付けてください…'), { target: { value: 'APIを説明します。' } });
    fireEvent.click(screen.getByRole('button', { name: /クイック生成/ }));
    await waitFor(() => expect(api.quickCreate).toHaveBeenCalledTimes(1));
    const body = vi.mocked(api.quickCreate).mock.calls[0][0];
    expect(body.voicevox_speed_scale).toBe(1);
    expect(body).not.toHaveProperty('voicevox_speaker_id');
    expect(body).not.toHaveProperty('narration_sentence_pause_seconds');
  });
});
