/** Route-level quick workflow: paste a script, queue it, and watch progress. */
import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { ProgressBar } from '@/components/ProgressBar';
import { StatusBadge } from '@/components/StatusBadge';
import { DEFAULT_PACING, PacingSettings, type Pacing } from '@/components/PacingSettings';
import { useLiveProject, useQuickCreate } from '@/api/hooks';

const STAGE_LABELS: Record<string, string> = {
  split: '台本を分割中',
  plan: '画面構成を設計中',
  image: 'スライド・図を描画中',
  audio: '音声を合成中',
  render: '動画を書き出し中',
  concat: '動画を連結中',
  queued: '順番待ち',
};

/**
 * Paste a script, press one button, get an MP4.
 *
 * Everything else (title, block split, visuals, narration, encoding) is
 * decided downstream, so the only input this page needs is the script.
 */
export function QuickGeneratePage() {
  const [script, setScript] = useState('');
  const [pacing, setPacing] = useState<Pacing>(DEFAULT_PACING);
  const [projectId, setProjectId] = useState<number | null>(null);
  const quick = useQuickCreate();
  const { data: project } = useLiveProject(projectId);

  const running = projectId != null && project?.status !== 'completed' && project?.status !== 'failed';
  const done = project?.status === 'completed' && Boolean(project?.output_video_path);

  const start = () => {
    if (!script.trim()) return;
    quick.mutate(
      { source_script: script, ...pacing },
      { onSuccess: (res) => setProjectId(res.project.id) },
    );
  };

  const reset = () => {
    setProjectId(null);
    setScript('');
    // Pacing is deliberately kept: someone who slowed the narration down once
    // usually wants the next video the same way.
    quick.reset();
  };

  return (
    <Layout>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-800">クイック生成</h1>
        <Link to="/projects" className="text-sm text-slate-500 hover:underline">
          プロジェクト一覧
        </Link>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        台本を貼り付けてボタンを押すだけ。タイトル・分割・図・音声・動画まで自動で作ります。
      </p>

      {projectId == null && (
        <div className="mt-6 space-y-4">
          <textarea
            className="h-96 w-full rounded-lg border border-slate-300 p-4 font-mono text-sm leading-relaxed
                       focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500"
            placeholder="ここに日本語の台本を貼り付けてください…"
            value={script}
            onChange={(e) => setScript(e.target.value)}
            disabled={quick.isPending}
          />
          <PacingSettings value={pacing} onChange={setPacing} disabled={quick.isPending} />
          <div className="flex items-center gap-4">
            <button
              className="btn-primary px-6 py-3 text-base disabled:cursor-not-allowed disabled:opacity-50"
              onClick={start}
              disabled={!script.trim() || quick.isPending}
            >
              {quick.isPending ? '開始しています…' : '⚡ クイック生成'}
            </button>
            <span className="text-xs text-slate-500">{script.length.toLocaleString()} 文字</span>
          </div>
          {quick.error && (
            <p className="text-sm text-red-600">
              開始に失敗しました: {(quick.error as Error).message}
            </p>
          )}
        </div>
      )}

      {projectId != null && (
        <div className="mt-6 space-y-5">
          <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-slate-800">
                  {project?.title ?? 'タイトルを生成中…'}
                </h2>
                <p className="mt-1 text-xs text-slate-500">{project?.block_count ?? 0} ブロック</p>
              </div>
              {project && <StatusBadge status={project.status} />}
            </div>

            <div className="mt-4">
              <ProgressBar
                progress={project?.progress ?? 0}
                stage={
                  project?.current_stage
                    ? STAGE_LABELS[project.current_stage] ?? project.current_stage
                    : null
                }
              />
            </div>

            {project?.error_message && (
              <p className="mt-3 rounded bg-red-50 px-3 py-2 text-sm text-red-700">
                {project.error_message}
              </p>
            )}
          </div>

          {done && (
            <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
              <h3 className="mb-3 text-sm font-semibold text-slate-700">完成しました</h3>
              <video
                className="w-full rounded-md bg-black"
                controls
                src={`/api/projects/${projectId}/download`}
              />
              <div className="mt-4 flex flex-wrap gap-3">
                <a className="btn-primary" href={`/api/projects/${projectId}/download`} download>
                  MP4 をダウンロード
                </a>
                <Link className="btn-secondary" to={`/projects/${projectId}`}>
                  ブロックを確認・修正
                </Link>
                <button className="btn-secondary" onClick={reset}>
                  別の台本で作る
                </button>
              </div>
            </div>
          )}

          {running && (
            <p className="text-sm text-slate-500">
              生成中です。このページを開いたままお待ちください（長い台本では数分かかります）。
            </p>
          )}
        </div>
      )}
    </Layout>
  );
}
