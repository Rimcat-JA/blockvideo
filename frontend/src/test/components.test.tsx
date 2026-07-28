import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from '@/components/StatusBadge';

describe('StatusBadge', () => {
  it('renders the status text', () => {
    render(<StatusBadge status="completed" />);
    expect(screen.getByText('completed')).toBeInTheDocument();
  });

  it('falls back gracefully for unknown statuses', () => {
    render(<StatusBadge status="weird-status" />);
    expect(screen.getByText('weird-status')).toBeInTheDocument();
  });
});