/** Provisional observations are separate from immutable operation receipts. */
import { useEffect, useState } from 'react';
import { api } from '@/api/client';
import type { CandidateReadinessSnapshot } from '@/lib/language-types';
import { operationNames } from '@/lib/language-results';
import type { ProjectOperationId } from '@/lib/types';

const reasons: Record<string, string> = {
  project_busy: '生成中のため変更できません',
  arguments_unchecked: '値・参照を確認してから判定します',
  target_required: '対象の選択が必要です',
  target_not_found: '対象が見つかりません',
  operation_not_found: 'この版の操作は現在登録されていません',
};

export function LanguageCandidateReadiness({ requestId, recorded }: {
  requestId: string; recorded: CandidateReadinessSnapshot;
}) {
  const [refresh, setRefresh] = useState(0);
  const [current, setCurrent] = useState<CandidateReadinessSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!refresh) return;
    const controller = new AbortController();
    api.getCandidateReadiness(requestId, controller.signal).then(value => {
      if (!controller.signal.aborted) { setCurrent(value); setFailed(false); }
    }).catch(() => { if (!controller.signal.aborted) setFailed(true); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [requestId, refresh]);
  const snapshot = current ?? recorded;
  return <details className="mt-3 rounded border border-slate-200 p-3 text-sm text-slate-700">
    <summary className="cursor-pointer font-medium">候補の実行条件</summary>
    <p className="mt-2">{current ? '状態確認時点の候補' : '依頼を判断した時点の候補'}（{new Date(snapshot.observed_at * 1000).toLocaleTimeString('ja-JP')}）</p>
    <p className="mt-1 text-xs text-slate-500">入力値を確定する前の暫定情報です。実行時に対象・値・最新状態をもう一度確認します。</p>
    <ul className="mt-2 space-y-2 break-words">{snapshot.candidates.map(c => <li key={`${c.operation_id}@${c.operation_version}`}>
      <span className="font-medium">{operationNames[c.operation_id as ProjectOperationId] ?? c.operation_id}（版{c.operation_version}）</span>
      <span className={`block ${c.readiness === 'blocked' ? 'text-amber-800' : ''}`}>
        {c.reason_code ? reasons[c.reason_code] : '対象・状態の制約なし（暫定）'}
      </span>
    </li>)}</ul>
    {failed && <p className="mt-2 text-red-700" role="alert">現在の状態を取得できませんでした。表示中の情報は前回の確認時点のものです。</p>}
    <button type="button" className="btn-secondary mt-3" disabled={loading} onClick={() => {
      setLoading(true); setFailed(false); setRefresh(value => value + 1);
    }}>{loading ? '状態を確認中…' : '候補の状態を更新'}</button>
    <p className="mt-2 text-xs text-slate-500">状態の更新だけでは操作を実行しません。実行したい場合は、最新の状態で新しく依頼してください。</p>
  </details>;
}
