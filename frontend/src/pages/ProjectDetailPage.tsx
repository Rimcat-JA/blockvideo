/** Project settings, immutable output history and durable generation controls. */
import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { useProject, useProjectBlocks, useProjectHistory } from '@/api/hooks';
import { useProjectOperation } from '@/api/useProjectOperation';
import { useLanguageRequest } from '@/api/useLanguageRequest';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/api/client';
import { LanguagePanel } from '@/components/LanguagePanel';
import { LanguageConnection } from '@/components/LanguageConnection';
import { LanguageTargetSelect } from '@/components/LanguageTargetSelect';
import { ProgressBar } from '@/components/ProgressBar';
import { StatusBadge } from '@/components/StatusBadge';
import { BlockList } from '@/components/BlockList';
import { ProjectSettingsEditor } from '@/components/ProjectSettingsEditor';
import { ProjectOutputHistory } from '@/components/ProjectOutputHistory';
import { SettingsHistory } from '@/components/SettingsHistory';
import { GenerationHistory } from '@/components/GenerationHistory';

export function ProjectDetailPage() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  return <ProjectMonitor key={id} id={id} />;
}

function ProjectMonitor({ id }: { id: number }) {
  const project = useProject(id);
  const blocks = useProjectBlocks(id);
  const history = useProjectHistory(id);
  const command = useProjectOperation(id);
  const language = useLanguageRequest(id);
  // Same cached query as the connection panel; YOLO is offered only if the server allows it.
  const connection = useQuery({ queryKey: ['language-connection'], queryFn: () => api.languageConnection(),
    retry: false, staleTime: Infinity, refetchOnWindowFocus: false });
  const yoloAvailable = connection.data?.yolo_enabled === true;
  const actionRevalidation = useRef(false);
  const [revalidatingAction, setRevalidatingAction] = useState(false);
  const [actionRefreshRequired, setActionRefreshRequired] = useState(false);
  const jobs = history.data?.jobs ?? [];
  const active = jobs.filter((job) => job.status === 'pending' || job.status === 'running');
  const projectRecovery = project.data?.generation_recovery;
  const projectBusy = projectRecovery?.code === 'busy'
    && projectRecovery.recommended_action === 'wait';
  const projectUnknown = projectRecovery?.code === 'external_outcome_unknown'
    && projectRecovery.recommended_action === 'check_provider';
  const projectReady = projectRecovery?.code === 'ready'
    && projectRecovery.recommended_action === 'generate';
  const running = projectBusy || active.length > 0
    || ['splitting', 'planning', 'generating', 'rendering'].includes(project.data?.status ?? '');
  const unknown = projectUnknown || jobs.some((job) => job.recommended_action === 'check_provider');
  const viewsDiffer = history.data != null && project.data != null && history.data.revision !== project.data.revision;
  const actionsUnavailable = command.locked || language.locked || project.isFetching || history.isFetching
    || revalidatingAction || !history.data || !!history.error || viewsDiffer;
  const blocked = running || actionsUnavailable;
  const { refetch: refreshProject } = project;
  const { refetch: refreshBlocks } = blocks;
  const { refetch: refreshHistory } = history;
  const refreshState = () => {
    setActionRefreshRequired(false);
    return Promise.all([refreshProject(), refreshBlocks(), refreshHistory()]);
  };
  const revalidateRecoveryAction = async (action: 'retry' | 'cancel', jobId: number) => {
    if (actionRevalidation.current || command.locked) return;
    actionRevalidation.current = true;
    setRevalidatingAction(true);
    setActionRefreshRequired(false);
    try {
      const [currentProject, currentHistory] = await Promise.all([refreshProject(), refreshHistory()]);
      const currentJob = currentHistory.data?.jobs.find((job) => job.id === jobId);
      const coherent = !currentProject.isError && !currentHistory.isError
        && currentProject.data != null && currentHistory.data != null
        && currentProject.data.revision === currentHistory.data.revision;
      const currentRecovery = currentProject.data?.generation_recovery;
      const authorized = action === 'retry'
        ? currentJob != null
          && currentRecovery?.code === 'ready'
          && currentRecovery.recommended_action === 'generate'
          && (currentJob.status === 'failed' || currentJob.status === 'cancelled')
          && currentJob.recommended_action === 'retry_current'
          && currentJob.retryable === true
        : currentJob != null
          && currentRecovery?.code === 'busy'
          && currentRecovery.recommended_action === 'wait'
          && (currentJob.status === 'pending' || currentJob.status === 'running')
          && currentJob.recommended_action === 'wait'
          && currentJob.retryable === false
          && !currentJob.cancel_requested;
      if (!coherent || !authorized || currentProject.data == null) {
        setActionRefreshRequired(true);
        return;
      }
      if (action === 'retry') {
        command.execute('project.generation.retry', currentProject.data.revision, { job_id: jobId }, '現在の設定での再実行');
      } else {
        command.execute('project.generation.cancel', currentProject.data.revision, { job_id: jobId }, 'キャンセル要求');
      }
    } catch {
      setActionRefreshRequired(true);
    } finally {
      actionRevalidation.current = false;
      setRevalidatingAction(false);
    }
  };
  const observedRevision = history.data?.revision;
  const observedJobState = jobs.map((job) => `${job.id}:${job.status}:${job.cancel_requested}`).join(',');

  useEffect(() => {
    void refreshProject();
    void refreshBlocks();
  }, [observedRevision, observedJobState, refreshProject, refreshBlocks]);

  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => { void refreshProject(); void refreshBlocks(); }, 2000);
    return () => clearInterval(timer);
  }, [running, refreshProject, refreshBlocks]);

  if (project.isLoading) return <Layout><p className="text-sm text-slate-500">読み込み中...</p></Layout>;
  if (project.error || !project.data) return <Layout><p className="text-sm text-red-600">
    プロジェクトの読み込みに失敗しました: {project.error?.message}
  </p></Layout>;

  const p = project.data;
  const revision = p.revision;
  return (
    <Layout>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link to="/" className="text-xs text-slate-500 hover:underline">← 一覧へ戻る</Link>
          <h1 className="mt-1 text-2xl font-bold text-slate-800">{p.title}</h1>
          <p className="text-sm text-slate-500">設定の版 {revision} · {p.block_count} ブロック</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={p.status} />
          <button type="button" className="btn-primary" disabled={blocked || !projectReady}
            onClick={() => command.execute('project.generation.start', revision, {}, '生成開始')}>
            {p.block_count > 0 ? '現在の設定で新しく生成' : '生成開始'}
          </button>
          <button type="button" className="btn-secondary" disabled={blocked || !projectReady || !p.block_count}
            onClick={() => command.execute('project.generation.start', revision, { kind: 'rerender' }, '動画の再レンダリング')}>レンダリングのみ再実行</button>
          {active[0] && <button type="button" className="btn-danger" disabled={actionsUnavailable || !projectBusy || active[0].cancel_requested}
            onClick={() => { void revalidateRecoveryAction('cancel', active[0].id); }}>
            {active[0].cancel_requested ? 'キャンセル要求済み' : '生成をキャンセル'}
          </button>}
        </div>
      </div>
      <LanguageTargetSelect projectId={id} />
      <LanguageConnection />
      <LanguagePanel project={p} history={history.data} controller={language} yoloAvailable={yoloAvailable}
        disabled={command.locked} unavailable={!history.data || !!history.error || viewsDiffer} running={running} />
      <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
        <ProgressBar progress={p.progress} stage={p.current_stage} status={p.status} />
      </div>
      {running && <p className="mt-4 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800" role="status">
        {projectBusy
          ? '生成処理が完了するまでお待ちください。停止要求後も、安全に停止するまで生成操作はできません。'
          : '生成中のため設定を変更できません。キャンセルを要求した場合も、停止完了までお待ちください。'}
      </p>}
      {unknown && <p className="mt-4 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800" role="status">
        外部処理の結果が未確定です。自動で再送せず、外部サービス側の実行履歴を確認してください。
      </p>}
      {history.error && <p role="alert" className="mt-4 text-sm text-red-600">履歴を取得できません。状態を確認できるまで操作を待機します。</p>}
      {viewsDiffer && <div role="status" className="mt-4 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
        <p>プロジェクトと生成履歴の版が一致しません。最新の状態を取得するまで操作できません。</p>
        <button type="button" className="btn-secondary mt-2" onClick={() => { void refreshState(); }}>最新の状態を再取得</button>
      </div>}
      {actionRefreshRequired && <p role="status" className="mt-4 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
        状態が変わりました。最新の状態を確認してから、表示された操作を選び直してください。
      </p>}
      {p.error_message && <p role="alert" className="mt-4 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        {p.error_message}
      </p>}
      {command.error && <p role="alert" className="mt-4 text-sm text-red-600">{command.error}</p>}
      {command.message && <p role="status" className="mt-4 text-sm text-green-700">{command.message}</p>}
      {command.pending && !command.isSending && <div className="mt-4 space-y-2 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
        <p>「{command.pending.label}」の応答が未確認です。同じ要求を再送して結果を確認できます。二重には実行されません。</p>
        <button type="button" className="btn-secondary" onClick={command.resend}>同じ要求を再送する</button>
      </div>}
      {history.data && <ProjectOutputHistory projectId={id} history={history.data} legacyVideo={!!p.output_video_path}
        disabled={blocked} onRestore={(artifactId) => command.execute('project.artifact.restore', revision, { artifact_id: artifactId }, '完成動画を戻す')} />}
      <ProjectSettingsEditor key={`${id}:${p.revision}`} project={p} disabled={blocked}
        onSave={(changes) => command.execute('project.settings.update', p.revision, changes, '設定保存')} />
      {history.data && <>
        <SettingsHistory versions={history.data.settings_versions} revision={revision} disabled={blocked}
          onRestore={(selectedRevision) => command.execute('project.settings.restore', revision, { revision: selectedRevision }, '設定の復元')} />
        <GenerationHistory jobs={jobs} running={running} disabled={actionsUnavailable}
          retryBlocked={!projectReady} cancelBlocked={!projectBusy}
          onCancel={(jobId) => { void revalidateRecoveryAction('cancel', jobId); }}
          onRetry={(jobId) => { void revalidateRecoveryAction('retry', jobId); }} />
      </>}
      <section className="mt-6">
        <h2 className="text-base font-semibold text-slate-800">ブロック ({blocks.data?.length ?? 0})</h2>
        <div className="mt-3"><BlockList blocks={blocks.data ?? []} disabled={blocked || !projectReady}
          onGenerate={(kind, blockIndex) => command.execute('project.generation.start', revision,
            kind === 'rerender' ? { kind } : { kind, block_index: blockIndex },
            kind === 'block_audio' ? `ブロック ${blockIndex} の音声再生成` : kind === 'block_visual' ? `ブロック ${blockIndex} の画像再生成` : '動画の再レンダリング')} /></div>
      </section>
    </Layout>
  );
}
