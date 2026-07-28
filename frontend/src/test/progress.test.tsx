import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ProgressBar } from '@/components/ProgressBar';

describe('ProgressBar', () => {
  it('shows percent', () => {
    render(<ProgressBar progress={0.42} stage="image" status="running" />);
    expect(screen.getByText('42%')).toBeInTheDocument();
    expect(screen.getByText(/工程: image/)).toBeInTheDocument();
  });

  it('clamps out-of-range values', () => {
    render(<ProgressBar progress={2} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('handles missing stage gracefully', () => {
    render(<ProgressBar progress={0} />);
    expect(screen.getByText('待機中')).toBeInTheDocument();
  });
});