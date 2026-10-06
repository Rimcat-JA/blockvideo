import type { JobSummary, RecoveryCode, RecommendedAction } from '@/lib/types';

const recoveryLabels: Record<RecoveryCode, string> = {
  wait: '処理の完了待ち',
  safe_retry: '再実行できます',
  external_outcome_unknown: '外部処理の結果が不明',
  refresh_required: '状態の再取得が必要',
  cancelled: 'キャンセル完了',
  completed: '生成完了',
  failed: '生成失敗',
};

const actionGuidance: Record<RecommendedAction, string | null> = {
  wait: '処理が完了するまでお待ちください。',
  retry_current: '現在の設定で新しい生成として再実行できます。',
  check_provider: '自動で再送しません。外部サービス側の実行履歴を確認してください。',
  refresh: '画面を再読み込みして最新の状態を確認してください。',
  none: null,
};

export function RecoveryStatus({ job }: { job: JobSummary }) {
  const guidance = actionGuidance[job.recommended_action];

  return (
    <div role="status" className="mt-2 text-sm text-slate-600">
      <p className="font-medium text-slate-700">{recoveryLabels[job.recovery_code]}</p>
      {guidance && <p className="mt-1">{guidance}</p>}
    </div>
  );
}
