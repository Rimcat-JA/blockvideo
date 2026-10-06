/** Successful videos are immutable history; their relation to saved settings is explicit. */
import { api } from '@/api/client';
import type { ProjectHistory } from '@/lib/types';

export function ProjectOutputHistory({ projectId, history, legacyVideo, onRestore, disabled = false }: {
  projectId: number; history: ProjectHistory; legacyVideo: boolean;
  onRestore?: (artifactId: number) => void; disabled?: boolean;
}) {
  const latest = history.artifacts.find((artifact) => artifact.id === history.current_artifact_id) ?? history.artifacts[0];
  const showLegacy = !latest && legacyVideo && history.current_artifact_id == null && history.output_state !== 'missing';
  const stateLabel = showLegacy ? '設定の対応未確認の以前の動画' : {
    none: '完成動画はまだありません', current: '最新設定の動画',
    stale: '設定は保存済み・動画には未反映', missing: '完成動画のファイルが見つかりません',
  }[history.output_state];

  return (
    <section className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-semibold text-slate-800">完成動画</h2>
      <p className={`mt-2 text-sm ${history.output_state === 'current' ? 'text-green-700' : 'text-amber-700'}`} role="status">{stateLabel}</p>
      {latest?.available && <video controls className="mt-3 w-full max-w-3xl rounded border border-slate-200" src={latest.video_url} />}
      {showLegacy && <>
        <video controls className="mt-3 w-full max-w-3xl rounded border border-slate-200" src={api.downloadUrl(projectId)} />
      </>}
      <h3 className="mt-5 text-sm font-semibold text-slate-800">成功した動画の履歴（{history.artifacts.length}件）</h3>
      <p className="mt-1 text-xs text-slate-500">新しい生成に成功しても、以前の動画は残ります。</p>
      <ul className="mt-3 divide-y divide-slate-100">
        {history.artifacts.map((artifact) => (
          <li key={artifact.id} className="flex flex-wrap items-center justify-between gap-3 py-3 text-sm">
            <div>
              <span className={artifact.is_current ? 'badge-completed' : 'badge-pending'}>{artifact.is_current ? '最新設定' : '旧版'}</span>
              <span className="ml-2">{artifact.revision == null ? '設定の版は不明' : `設定の版 ${artifact.revision}`}</span>
              <time className="ml-2 text-slate-500" dateTime={artifact.created_at}>{new Date(artifact.created_at).toLocaleString('ja-JP')}</time>
            </div>
            {artifact.available ? <div className="flex gap-2">
              <a className="btn-secondary" href={artifact.video_url}>動画 {artifact.id} を開く・保存</a>
              {artifact.subtitle_url && <a className="btn-secondary" href={artifact.subtitle_url}>字幕</a>}
              {onRestore && artifact.id !== history.current_artifact_id && <button type="button" className="btn-secondary"
                disabled={disabled} onClick={() => onRestore(artifact.id)}>動画 {artifact.id} を現在の動画に戻す</button>}
            </div> : <span className="text-red-600">ファイルが見つかりません</span>}
          </li>
        ))}
      </ul>
    </section>
  );
}
