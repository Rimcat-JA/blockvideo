/** What an unattended (YOLO) request did on the user's behalf. */
import type { LanguageResponse, YoloGuardCode } from '@/lib/language-types';

const guardLabels: Record<YoloGuardCode, string> = {
  reference: 'ジョブ番号・版番号を推測しました',
  subtitle_value: '字幕サイズの値を推測しました',
  settings_value: '設定の数値を推測しました',
  pending_settings: '前の依頼の設定を補わずに進めました',
  reading: '用語の読み方を推測しました',
  empty_settings: '変更内容が空のまま進めました',
};

const operationLabels: Record<string, string> = {
  'project.generation.start': '動画の生成を開始',
  'project.generation.retry': 'ジョブの再試行',
  'project.generation.cancel': 'ジョブのキャンセル',
  'project.settings.update': '設定の保存',
  'project.settings.restore': '設定の版を戻す',
  'project.artifact.restore': '完成動画を戻す',
  'project.subtitle-font-size.set': '字幕サイズの設定',
  'project.subtitle-font-size.adjust': '字幕サイズの増減',
  'project.status.get': '状態の確認',
};

export function LanguageYoloReport({ response }: { response: LanguageResponse }) {
  const report = response.yolo_report;
  if (response.execution_mode !== 'yolo' || !report) return null;
  return <section className="mt-3 rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900" aria-label="自動実行の報告">
    <p className="font-semibold">確認なしで自動実行しました（YOLO）</p>
    {report.auto_confirmed.length > 0 && <p className="mt-1">自動で確認した操作：{report.auto_confirmed.map((id) => operationLabels[id] ?? id).join('、')}</p>}
    {report.bypassed_guards.length > 0 ? <ul className="mt-1 list-disc pl-5">
      {report.bypassed_guards.map((code) => <li key={code}>{guardLabels[code]}</li>)}
    </ul> : <p className="mt-1">依頼に書かれていない値の推測はありませんでした。</p>}
    {(report.dropped_steps ?? []).length > 0 && <p className="mt-1">否定された手順は実行しませんでした：{(report.dropped_steps ?? []).map((id) => operationLabels[id] ?? id).join('、')}</p>}
    {report.unresolved && <p className="mt-1">{report.unresolved}</p>}
    <p className="mt-1 text-xs">結果が意図と違う場合は、設定履歴から依頼前の版に戻せます。</p>
  </section>;
}
