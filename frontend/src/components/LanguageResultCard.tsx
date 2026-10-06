/** An authoritative result, proposal for review, or explicit non-execution state. */
import type { LanguageResponse } from '@/lib/language-types';
import type { ProjectDetail, ProjectHistory } from '@/lib/types';
import { failureMessage, generationKinds, operationNames, settingChanges } from '@/lib/language-results';
import { settingNames, settingValue } from '@/lib/settings-display';
import { LanguageVideoState } from '@/components/LanguageVideoState';
import { LanguageDiagnostics } from '@/components/LanguageDiagnostics';
import { LanguageCandidateReadiness } from '@/components/LanguageCandidateReadiness';

export function LanguageResultCard({ response, project, history, unavailable, busy, disabled, onConfirm, onEdit }: {
  response: LanguageResponse; project: ProjectDetail; history?: ProjectHistory;
  unavailable: boolean; busy: boolean; disabled: boolean; onConfirm: () => void; onEdit: () => void;
}) {
  const committed = response.executed && response.result != null;
  const prepared = response.generation_request ?? response.prepared_request;
  const operation = response.result?.operation_id ?? prepared?.operation_id;
  const generation = !!response.generation_request || ['project.generation.start', 'project.generation.retry'].includes(operation ?? '');
  const settings = operation?.startsWith('project.settings.') || operation?.startsWith('project.subtitle-font-size.');
  const changes = settingChanges(response, history);
  const preparedRevision = prepared?.base_revision ?? response.base_revision;
  const stale = preparedRevision !== project.revision;
  const proposal = response.interpretation?.proposal;
  const question = response.clarification?.question ?? (proposal?.kind === 'clarification' ? proposal.question : undefined);
  const failureCode = (response.failure ?? response.interpretation?.failure)?.reason_code;
  const candidateState = response.diagnostics?.retrieval?.stages.at(-1)?.candidate_state;
  const connectionFailure = ['model_not_configured', 'configuration_error', 'connection_failed', 'http_error', 'model_mismatch', 'timeout'].includes(failureCode ?? '');
  const headings: Record<LanguageResponse['status'], string> = {
    interpreting: '依頼の内容を確認しています', ready: generation ? '動画を生成する前に確認' : '操作内容の確認',
    needs_input: '追加の内容を教えてください', unsupported: 'この操作には対応していません',
    blocked: candidateState ? 'この依頼は実行しませんでした' : '現在は実行できません', error: '依頼を処理できませんでした', completed: '操作の結果',
    dismissed: 'この依頼では実行しません',
  };
  const title = committed && response.generate_after_save
    ? response.generation_result ? '設定を保存し、動画生成を受け付けました' : '設定は保存済み・動画生成は未開始'
    : committed ? settings ? response.result!.changed ? '設定を保存しました' : '設定の変更はありません'
    : generation ? '動画生成を受け付けました' : operation === 'project.generation.cancel' ? '停止要求を受け付けました' : '状態を確認しました'
    : headings[response.status];

  return <article className={`mt-4 rounded-lg border p-4 ${committed ? 'border-emerald-200 bg-white' : 'border-slate-200 bg-white'}`} aria-label="自然言語の結果">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h3 className="font-semibold text-slate-900" role="status">{title}</h3>
      <span className="text-xs text-slate-500">対象: {project.title} · #{project.id}</span>
    </div>
    {response.superseded_by && <p className="mt-2 text-sm text-amber-800">この依頼には後の回答・訂正があります。古い内容では実行できません。</p>}
    {!!response.diagnostics?.retrieval && (response.diagnostics.retrieval.expansion_count > 0 || response.diagnostics.retrieval.all_tools_count > 0) &&
      <p className="mt-2 text-sm text-slate-600">最初の候補では十分に判断できなかったため、確認する操作の範囲を広げました。</p>}
    {committed && <>
      <p className="mt-2 text-sm text-slate-600">{operationNames[operation!] ?? '操作'} · 設定の版 {response.result!.revision}</p>
      {response.result!.revision < project.revision && <p className="mt-1 text-sm text-amber-800">この結果の後に設定が変更されています。現在は版 {project.revision} です。</p>}
      {changes.length > 0 && <dl className="mt-3 grid gap-2 sm:grid-cols-2" aria-label="保存された変更">
        {changes.map(({ field, before, after }) => <div key={field} className="rounded bg-slate-50 p-3">
          <dt className="text-xs text-slate-500">{settingNames[field] ?? field}</dt>
          <dd className="mt-1 break-words font-medium text-slate-800">{before === undefined ? '変更前は未取得' : settingValue(field, before)} → {settingValue(field, after)}</dd>
        </div>)}
      </dl>}
      {operation === 'project.status.get' && <p className="mt-2 text-sm text-slate-700">
        確認時の字幕サイズ: {String(response.result!.data.subtitle_font_size ?? '不明')}px
      </p>}
      {settings && !response.generation_result && <p className="mt-3 text-sm text-slate-600">{response.generate_after_save ? '設定はまとめて保存しました。動画生成は開始していません。生成を取り下げても設定は残ります。' : 'この設定変更では動画生成を開始していません。'}</p>}
      {settings && response.result!.changed && <a href="#settings-history" className="mt-2 inline-block text-sm text-indigo-700 underline">設定を戻す（履歴から選ぶ）</a>}
    </>}
    {response.status === 'interpreting' && <p className="mt-2 text-sm text-slate-600">結果が確定するまでお待ちください。</p>}
    {response.status === 'ready' && prepared && <div className="mt-3 space-y-2 text-sm text-slate-700">
      <p>{operationNames[prepared.operation_id]} · 使用する設定の版 {preparedRevision}</p>
      {generation && <p>{prepared.operation_id === 'project.generation.retry' ? `ジョブ ${String(prepared.arguments.job_id)} を現在の設定で再試行`
        : generationKinds[String(prepared.arguments.kind ?? 'full')]}
        {prepared.arguments.block_index != null ? `（ブロック ${String(prepared.arguments.block_index)}）` : ''}</p>}
      {!generation && <ul className="space-y-1">{Object.entries(prepared.operation_version === 2 && prepared.operation_id === 'project.settings.update'
        ? { ...(prepared.arguments.settings as Record<string, unknown>), ...(prepared.arguments.subtitle_font_size_delta != null ? { delta: prepared.arguments.subtitle_font_size_delta } : {}) }
        : prepared.arguments).map(([field, value]) =>
        <li key={field}>{settingNames[field] ?? ({ value: '字幕サイズ', delta: '字幕サイズの増減', revision: '戻す設定の版', job_id: '対象ジョブ', artifact_id: '戻す完成動画の番号' } as Record<string, string>)[field] ?? field}: {settingValue(field === 'value' || field === 'delta' ? 'subtitle_font_size' : field, value)}</li>)}</ul>}
      <p>{generation ? 'まだ動画生成は開始していません。対象と内容を確認してください。' : settings ? 'まだ設定は変更していません。' : 'まだこの操作は実行していません。'}</p>
      {stale && <p className="text-amber-800">設定が変わったため、この提案は実行できません。最新の状態で依頼し直してください。</p>}
      {response.requires_confirmation && <button type="button" className="btn-primary" disabled={busy || disabled || stale || unavailable || !!response.superseded_by} onClick={onConfirm}>
        {generation ? 'この内容で動画生成を開始' : settings ? 'この内容で設定を保存' : 'この内容で操作を実行'}
      </button>}
      <button type="button" className="btn-secondary ml-2" disabled={busy} onClick={onEdit}>依頼文を編集する</button>
    </div>}
    {question && <div className="mt-3 text-sm text-slate-700">
      <p>{question}</p>
      <p className="mt-2 text-slate-500">{response.dialogue_available ? '上の回答欄から、質問に沿って回答してください。' : '必要な内容を含めて、依頼全体を入力し直してください。'}</p>
      <button type="button" className="btn-secondary mt-2" disabled={busy} onClick={onEdit}>依頼文を編集する</button>
    </div>}
    {response.status === 'unsupported' && proposal?.kind === 'unsupported' && <p className="mt-2 text-sm text-slate-600">{proposal.reason}</p>}
    {response.status === 'dismissed' && <p className="mt-2 text-sm text-slate-600">{proposal?.kind === 'no_operation' ? proposal.reason : '未実行の依頼を取り下げました。'} 保存済みの設定や進行中の動画は変更していません。</p>}
    {failureMessage(response) && <p className="mt-2 text-sm text-red-700" role="alert">{candidateState && response.status === 'blocked' ? '依頼の判断時点の理由：' : ''}{failureMessage(response)}</p>}
    {response.status === 'blocked' && failureCode === 'project_busy' &&
      <p className="mt-2 text-sm text-slate-600">実行する場合は、生成の完了を確認してから、新しく依頼してください。この変更は予約されていません。</p>}
    {response.status === 'error' && <div className="mt-2 text-sm text-slate-600">
      <p>{connectionFailure ? '接続先やモデルの状態を確認した後、依頼文を確認して新しく送信してください。'
        : '表示された理由と指定した値を確認し、依頼文を見直して新しく送信してください。'}保存済みの同じ要求を再送しても、このエラーの記録が返ります。</p>
      <button type="button" className="btn-secondary mt-2" disabled={busy} onClick={onEdit}>依頼文を確認してやり直す</button>
    </div>}
    {['needs_input', 'unsupported', 'blocked', 'error'].includes(response.status) && <p className="mt-2 text-sm text-slate-500">{committed ? '設定の保存は完了しています。後続の動画生成は開始していません。' : 'この依頼による設定変更・動画生成は行っていません。'}</p>}
    {committed && <LanguageVideoState response={response} project={project} history={history} unavailable={unavailable} />}
    {candidateState && <LanguageCandidateReadiness key={response.request_id} requestId={response.request_id} recorded={candidateState} />}
    <LanguageDiagnostics response={response} />
  </article>;
}
