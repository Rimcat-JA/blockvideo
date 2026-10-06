/** Video progress and reflection are independent from operation acknowledgement. */
import type { LanguageResponse } from '@/lib/language-types';
import type { ProjectDetail, ProjectHistory } from '@/lib/types';

export function LanguageVideoState({ response, project, history, unavailable }: {
  response: LanguageResponse; project: ProjectDetail; history?: ProjectHistory; unavailable: boolean;
}) {
  const expectedRevision = Math.max(project.revision, response.result?.revision ?? response.base_revision ?? 1);
  const verified = !unavailable && history && history.revision >= expectedRevision && history.revision === project.revision;
  const jobId = response.generation_result?.job_id ?? response.result?.job_id ?? (typeof response.result?.data.job_id === 'number' ? response.result.data.job_id : null);
  const job = verified ? history.jobs.find((item) => item.id === jobId) : undefined;
  const artifact = verified ? history.artifacts.find((item) => item.job_id === jobId && item.available) : undefined;
  const labels = {
    none: '完成動画はまだありません', current: '最新の設定が動画に反映されています',
    stale: '設定は保存済み・動画には未反映', missing: '完成動画のファイルを確認できません',
  };
  const jobLabels: Record<string, string> = {
    pending: '動画生成を受け付けました・開始待ち', running: '動画を生成中',
    failed: '動画生成に失敗しました', cancelled: '動画生成は停止しました',
    unknown: '外部処理の結果を確認できません',
  };
  return <div className="mt-4 rounded-md bg-slate-50 p-3" aria-label="動画への反映状況">
    <h4 className="text-sm font-semibold text-slate-800">動画への反映</h4>
    <p className={`mt-1 text-sm ${verified && history.output_state === 'current' ? 'text-emerald-700' : 'text-amber-800'}`}>
      {verified ? labels[history.output_state] : '最新の動画状態を確認中です'}
    </p>
    {jobId != null && <div className="mt-2 text-sm text-slate-700">
      <p>{!job ? `生成処理 ${jobId} の状態を確認中です` : job.status === 'completed'
        ? artifact ? 'この要求の動画が完成しました' : '生成処理は完了しました。動画ファイルを確認中です'
        : jobLabels[job.status] ?? '生成状態を確認中です'}</p>
      {job && ['pending', 'running'].includes(job.status) && <>
        <progress className="mt-2 h-2 w-full accent-indigo-600" aria-label="動画生成の進捗" max={1} value={job.progress} />
        <p>{Math.round(job.progress * 100)}%{job.cancel_requested ? ' · 停止できる区切りを待っています' : ''}</p>
      </>}
      {job?.error_message && <p className="mt-1 break-words text-red-700">{job.error_message}</p>}
      {artifact && <a href={artifact.video_url} className="mt-2 inline-block text-indigo-700 underline">この要求で生成した動画を開く</a>}
      {job && ['failed', 'cancelled', 'unknown'].includes(job.status) &&
        <a className="mt-2 inline-block text-indigo-700 underline" href="#generation-history">生成の履歴を確認する</a>}
    </div>}
  </div>;
}
