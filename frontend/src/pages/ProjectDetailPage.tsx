/** Route-level project monitor with controls, output preview, and block list. */
import { useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import {
  useCancel,
  useGenerateAll,
  useProject,
  useProjectBlocks,
  useRerender,
} from '@/api/hooks';
import { api } from '@/api/client';
import { ProgressBar } from '@/components/ProgressBar';
import { StatusBadge } from '@/components/StatusBadge';
import { BlockList } from '@/components/BlockList';

const POLL_INTERVAL_MS = 2000;

export function ProjectDetailPage() {
  /** Poll active projects and expose generation, cancellation, and rerender UI. */
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  const project = useProject(id);
  const blocks = useProjectBlocks(id);
  const generate = useGenerateAll(id);
  const cancel = useCancel(id);
  const rerender = useRerender(id);

  const isRunning = (s: string | undefined | null) =>
    !!s && !['completed', 'failed', 'cancelled'].includes(s);

  // Poll project + blocks while running
  useEffect(() => {
    if (!project.data) return;
    if (!isRunning(project.data.status)) return;
    const t = setInterval(() => {
      project.refetch();
      blocks.refetch();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.data?.status, id]);

  if (project.isLoading) {
    return (
      <Layout>
        <p className="text-sm text-slate-500">読み込み中...</p>
      </Layout>
    );
  }
  if (project.error || !project.data) {
    return (
      <Layout>
        <p className="text-sm text-red-600">
          プロジェクトの読み込みに失敗しました: {(project.error as Error | undefined)?.message}
        </p>
      </Layout>
    );
  }

  const p = project.data;
  const running = isRunning(p.status);
  return (
    <Layout>
      <div className="flex items-center justify-between">
        <div>
          <Link to="/" className="text-xs text-slate-500 hover:underline">
            ← 一覧へ戻る
          </Link>
          <h1 className="mt-1 text-2xl font-bold text-slate-800">{p.title}</h1>
          <p className="text-sm text-slate-500">
            {p.block_count} blocks · VOICEVOX: {p.voicevox_url} speaker={p.voicevox_speaker_id}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={p.status} />
          {!running ? (
            <button
              type="button"
              className="btn-primary"
              disabled={generate.isPending}
              onClick={() => generate.mutate()}
            >
              {p.block_count > 0 ? '再生成' : '生成開始'}
            </button>
          ) : (
            <button
              type="button"
              className="btn-danger"
              disabled={cancel.isPending}
              onClick={() => cancel.mutate()}
            >
              キャンセル
            </button>
          )}
          <button
            type="button"
            className="btn-secondary"
            disabled={rerender.isPending || !p.block_count}
            onClick={() => rerender.mutate()}
          >
            レンダリングのみ再実行
          </button>
        </div>
      </div>

      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
        <ProgressBar progress={p.progress} stage={p.current_stage} status={p.status} />
      </div>

      {p.error_message && (
        <div className="mt-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <strong className="font-semibold">エラー:</strong> {p.error_message}
        </div>
      )}

      {p.output_video_path && (
        <section className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
          <h2 className="text-base font-semibold text-slate-800">完成動画</h2>
          <video
            controls
            className="mt-3 w-full max-w-3xl rounded border border-slate-200"
            src={api.downloadUrl(id)}
          />
          <div className="mt-2 space-x-2">
            <a className="btn-secondary" href={api.downloadUrl(id)}>
              MP4をダウンロード
            </a>
            <a
              className="btn-secondary"
              href={api.artifactUrl(id, 'video', 0)}
            >
              ブロック0 (参照用)
            </a>
          </div>
        </section>
      )}

      <section className="mt-6">
        <h2 className="text-base font-semibold text-slate-800">ブロック ({blocks.data?.length ?? 0})</h2>
        <div className="mt-3">
          <BlockList projectId={id} blocks={blocks.data ?? []} />
        </div>
      </section>
    </Layout>
  );
}
