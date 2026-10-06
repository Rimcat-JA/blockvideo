/** Waiting for operation judgement/delivery is separate from rendering a video. */
import { useEffect, useState } from 'react';
import type { LanguageController } from '@/api/useLanguageRequest';

export function LanguageWaiting({ controller }: { controller: LanguageController }) {
  const { session, response, busy, issue, activity, activityStartedAt } = controller;
  const interpreting = response?.status === 'interpreting';
  const active = !issue && (busy || interpreting);
  const lookup = activity === 'lookup' && !interpreting;
  const confirming = session?.action === 'confirm';
  const started = lookup ? activityStartedAt
    : confirming ? session?.startedAt ?? activityStartedAt
    : response?.diagnostics?.started_at ? response.diagnostics.started_at * 1000
    : session?.startedAt ?? activityStartedAt;
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active, started]);
  if (!active) return null;
  const seconds = started == null ? null : Math.max(0, Math.floor((now - started) / 1000));
  return <div className="mt-3 rounded-lg border border-indigo-200 bg-white p-3 text-sm text-indigo-900" aria-label="操作の待ち時間">
    <p role="status">{lookup ? '保存済みの結果を照会中' : confirming ? '開始要求の結果を確認中' : '操作内容を判断・確認中'}
      {seconds != null && <span aria-live="off"> · {seconds}秒経過</span>}</p>
    <p className="mt-1 text-xs text-slate-600">{lookup ? '新しい操作は送信していません。'
      : confirming ? '動画生成の受付結果を確認しています。生成の進み具合は動画の状態に表示されます。'
      : '依頼の解釈・検証・保存結果を待っています。この待ち時間は動画生成の時間とは別です。'}</p>
    {seconds != null && seconds >= 30 && <p role="status" className="mt-2 text-amber-800">
      通常より時間がかかっています。処理は中止していません。同じ依頼を重複して送らず、このままお待ちください。
    </p>}
  </div>;
}
