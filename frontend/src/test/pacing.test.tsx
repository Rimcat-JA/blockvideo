/** PacingSettings defaults, sliders, reset, and speaker control tests. */
import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DEFAULT_PACING, PacingSettings, type Pacing } from '@/components/PacingSettings';

vi.mock('@/api/client', () => ({
  api: { speakers: () => Promise.resolve({ url: '', speakers: [] }) },
}));

function renderPanel(value: Pacing = DEFAULT_PACING, onChange = vi.fn()) {
  /** Render the settings panel inside the query provider used by its hook. */
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <PacingSettings value={value} onChange={onChange} />
    </QueryClientProvider>,
  );
  return onChange;
}

describe('PacingSettings', () => {
  it('shows the breath pause with its current value', () => {
    renderPanel();
    expect(screen.getByLabelText('文末の息継ぎ')).toHaveValue('1.5');
  });

  it('reports a changed pause without touching the other settings', () => {
    const onChange = renderPanel();
    fireEvent.change(screen.getByLabelText('文末の息継ぎ'), { target: { value: '2.4' } });
    expect(onChange).toHaveBeenCalledWith({
      ...DEFAULT_PACING,
      narration_sentence_pause_seconds: 2.4,
    });
  });

  it('marks itself as unmodified while every value is the default', () => {
    renderPanel();
    expect(screen.getByText('（既定のまま）')).toBeInTheDocument();
  });

  it('marks itself as modified once a value differs', () => {
    renderPanel({ ...DEFAULT_PACING, voicevox_speed_scale: 1.3 });
    expect(screen.getByText('（変更あり）')).toBeInTheDocument();
  });

  it('restores every default at once', () => {
    const onChange = renderPanel({ ...DEFAULT_PACING, post_margin_seconds: 4 });
    fireEvent.click(screen.getByText('既定値に戻す'));
    expect(onChange).toHaveBeenCalledWith(DEFAULT_PACING);
  });

  it('offers a pause of zero, which keeps VOICEVOX untouched', () => {
    renderPanel();
    expect(screen.getByLabelText('文末の息継ぎ')).toHaveAttribute('min', '0');
  });
});

describe('PacingSettings — slide cap', () => {
  it('defaults to one slide per block', () => {
    renderPanel();
    expect(screen.getByLabelText('1ブロックの最大スライド枚数')).toHaveValue('1');
  });

  it('never allows a block to show zero slides', () => {
    renderPanel();
    expect(screen.getByLabelText('1ブロックの最大スライド枚数')).toHaveAttribute('min', '1');
  });

  it('reports a changed cap as an integer', () => {
    const onChange = renderPanel();
    fireEvent.change(screen.getByLabelText('1ブロックの最大スライド枚数'), {
      target: { value: '5' },
    });
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_PACING, max_slides_per_block: 5 });
  });
});
