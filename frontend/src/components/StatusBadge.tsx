/** Small status-to-style presentation component for projects and blocks. */
import type { BlockStatus, ProjectStatus } from '@/lib/types';

const statusMap: Record<string, string> = {
  pending: 'badge-pending',
  running: 'badge-running',
  completed: 'badge-completed',
  failed: 'badge-failed',
  skipped: 'badge-pending',
  splitting: 'badge-running',
  planning: 'badge-running',
  generating: 'badge-running',
  rendering: 'badge-running',
  cancelled: 'badge-pending',
};

export function StatusBadge({ status }: { status: BlockStatus | ProjectStatus | string }) {
  /** Render a badge class chosen from the shared status map. */
  const cls = statusMap[status] ?? 'badge-pending';
  return <span className={cls}>{status}</span>;
}
