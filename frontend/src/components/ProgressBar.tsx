/** Progress display shared by quick generation and project detail pages. */
/** Props for a normalized 0..1 project progress display. */
interface ProgressBarProps {
  progress: number;
  stage?: string | null;
  status?: string | null;
}

export function ProgressBar({ progress, stage, status }: ProgressBarProps) {
  /** Clamp progress, format it as a percentage, and render the bar. */
  const clamped = Math.max(0, Math.min(1, progress || 0));
  const percent = Math.round(clamped * 100);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-slate-600">
        <span>
          {stage ? `工程: ${stage}` : '待機中'}
          {status ? ` (${status})` : ''}
        </span>
        <span>{percent}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
        <div
          className="h-full rounded-full bg-accent-600 transition-all"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
