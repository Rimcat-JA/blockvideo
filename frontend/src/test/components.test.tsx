/** Shared status badge rendering tests. */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from '@/components/StatusBadge';

describe('StatusBadge', () => {
  it('renders the status as a Japanese label', () => {
    render(<StatusBadge status="completed" />);
    expect(screen.getByText('完了')).toHaveAttribute('title', 'completed');
  });

  it('falls back gracefully for unknown statuses', () => {
    render(<StatusBadge status="weird-status" />);
    expect(screen.getByText('weird-status')).toBeInTheDocument();
  });
});
