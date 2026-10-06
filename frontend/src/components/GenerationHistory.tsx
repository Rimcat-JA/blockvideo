import { RecoveryStatus } from '@/components/RecoveryStatus';
import type { JobSummary } from '@/lib/types';
const stageLabels: Record<string, string> = { split: '台本分割', plan: '構成', image: '画像', audio: '音声', render: '動画組み立て', concat: '動画結合', subtitles: '字幕' };

export function GenerationHistory({
  jobs, disabled, running, retryBlocked = false, cancelBlocked = false, onRetry, onCancel,
}: {
  jobs: JobSummary[];
  disabled: boolean;
  running: boolean;
  retryBlocked?: boolean;
  cancelBlocked?: boolean;
  onRetry: (id: number) => void;
  onCancel: (id: number) => void;
}) {
  return (
    <section id="generation-history" className="mt-6 max-w-full rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-semibold text-slate-800">生成の履歴</h2>
      <p className="mt-1 text-sm text-slate-500">再実行は現在の設定で新しい生成を開始します。元の試行は履歴に残ります。</p>
      {!jobs.length && <p className="mt-3 text-sm text-slate-500">まだ生成していません。</p>}
      <ul className="mt-3 space-y-3">
        {jobs.map((job) => {
          const active = job.status === 'pending' || job.status === 'running';
          return <li key={job.id} className="min-w-0 rounded border border-slate-200 p-3 text-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="font-medium text-slate-800">生成 {job.id}
                {job.input_revision != null && <span className="ml-2 font-normal text-slate-500">設定の版 {job.input_revision}</span>}
              </p>
              {active ? <button type="button" className="btn-danger" disabled={disabled || cancelBlocked || job.cancel_requested}
                onClick={() => onCancel(job.id)}>{job.cancel_requested ? '停止を待っています' : 'キャンセルを要求'}</button>
                : job.recommended_action === 'retry_current' && job.retryable && <button type="button" className="btn-secondary"
                  disabled={disabled || running || retryBlocked} onClick={() => onRetry(job.id)}>現在の設定で再実行</button>}
            </div>
            <RecoveryStatus job={job} />
            {job.parent_job_id != null && <p className="mt-1 text-xs text-slate-500">生成 {job.parent_job_id} からの再実行</p>}
            {job.plan?.stages && <p className="mt-1 text-xs text-slate-500">実行する工程: {job.plan.stages.map((stage) => stageLabels[stage] ?? stage).join(' → ') || '変更なし'}</p>}
            {job.cancel_requested && active && <p className="mt-2 text-amber-700">キャンセル要求済み。処理が安全に区切れる所で停止します。停止完了まで設定は変更できません。</p>}
            {(job.recovery_message || job.error_message) && <p className="mt-2 break-words whitespace-pre-wrap text-slate-600">{job.recovery_message || job.error_message}</p>}
            {job.retry_blocked_reason && <p className="mt-1 break-words text-amber-700">{job.retry_blocked_reason}</p>}
          </li>;
        })}
      </ul>
    </section>
  );
}
