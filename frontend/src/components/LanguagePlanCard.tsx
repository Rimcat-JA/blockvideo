/** A multi-step plan: confirmed once, run step by step, stopped at the first failure. */
import type { LanguageResponse } from '@/lib/language-types';
import type { ProjectDetail } from '@/lib/types';
import { failureMessage, generationKinds, operationNames } from '@/lib/language-results';
import { settingNames, settingValue } from '@/lib/settings-display';

const GENERATION = ['project.generation.start', 'project.generation.retry'];

function describe(operationId: string, argumentsValue: Record<string, unknown>): string {
  if (operationId === 'project.generation.start') return generationKinds[String(argumentsValue.kind ?? 'full')] ?? '動画の生成';
  const entries = Object.entries(argumentsValue.settings && typeof argumentsValue.settings === 'object'
    ? argumentsValue.settings as Record<string, unknown> : argumentsValue);
  return entries.map(([field, value]) => `${settingNames[field] ?? ({ value: '字幕サイズ', delta: '字幕サイズの増減',
    revision: '戻す設定の版', job_id: '対象ジョブ', artifact_id: '戻す完成動画の番号' } as Record<string, string>)[field] ?? field}: ${settingValue(
    field === 'value' || field === 'delta' ? 'subtitle_font_size' : field, value)}`).join('、');
}

export function LanguagePlanCard({ response, project, busy, disabled, unavailable, onConfirm, onEdit }: {
  response: LanguageResponse; project: ProjectDetail; busy: boolean; disabled: boolean; unavailable: boolean;
  onConfirm: () => void; onEdit: () => void;
}) {
  const steps = response.plan ?? [];
  const done = response.plan_results ?? [];
  const generation = steps.some((step) => GENERATION.includes(step.operation_id));
  const stale = response.base_revision !== project.revision && done.length === 0;
  const title = response.status === 'completed' ? `${steps.length}件の手順をすべて実行しました`
    : response.status === 'blocked' && done.length > 0 ? `手順${done.length + 1}で停止しました`
    : response.status === 'ready' ? `${steps.length}件の手順を順に実行する前に確認` : '複数手順の依頼';
  return <article className="mt-4 rounded-lg border border-slate-200 bg-white p-4" aria-label="複数手順の結果">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <h3 className="font-semibold text-slate-900" role="status">{title}</h3>
      <span className="text-xs text-slate-500">対象: {project.title} · #{project.id}</span>
    </div>
    <ol className="mt-3 list-decimal space-y-1 pl-5 text-sm text-slate-700">
      {steps.map((step, index) => <li key={step.request_id ?? index}>
        <span className="font-medium">{operationNames[step.operation_id] ?? step.operation_id}</span>
        {describe(step.operation_id, step.arguments) && <span>（{describe(step.operation_id, step.arguments)}）</span>}
        <span className="ml-2 text-xs text-slate-500">{index < done.length ? `実行済み · 設定の版 ${done[index].revision}`
          : response.status === 'blocked' && index === done.length ? '失敗' : '未実行'}</span>
      </li>)}
    </ol>
    {response.status === 'ready' && <div className="mt-3 space-y-2 text-sm text-slate-700">
      <p>確認すると、上から順に実行します。途中で失敗した場合は、そこで止めて残りは実行しません。</p>
      {stale && <p className="text-amber-800">設定が変わったため、この計画は実行できません。最新の状態で依頼し直してください。</p>}
      {response.requires_confirmation && <button type="button" className="btn-primary"
        disabled={busy || disabled || stale || unavailable || !!response.superseded_by} onClick={onConfirm}>
        {generation ? 'この手順で実行し、動画生成も開始' : 'この手順で実行'}</button>}
      <button type="button" className="btn-secondary ml-2" disabled={busy} onClick={onEdit}>依頼文を編集する</button>
    </div>}
    {failureMessage(response) && <p className="mt-2 text-sm text-red-700" role="alert">{failureMessage(response)}</p>}
    {response.status === 'completed' && done.some((result) => result.changed) &&
      <a href="#settings-history" className="mt-2 inline-block text-sm text-indigo-700 underline">設定を戻す（履歴から選ぶ）</a>}
  </article>;
}
