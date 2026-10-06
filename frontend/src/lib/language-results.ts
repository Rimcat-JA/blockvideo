/** Present committed values, not statements made by the interpreter. */
import type { LanguageResponse } from '@/lib/language-types';
import type { ProjectHistory } from '@/lib/types';

export const operationNames: Record<string, string> = {
  'project.status.get': '状態の確認', 'project.subtitle-font-size.set': '字幕サイズの変更',
  'project.subtitle-font-size.adjust': '字幕サイズの調整', 'project.settings.update': '設定の変更',
  'project.settings.restore': '設定の復元', 'project.generation.start': '動画の生成',
  'project.generation.retry': '現在の設定で再試行', 'project.generation.cancel': '生成のキャンセル',
  'project.artifact.restore': '完成動画を戻す',
};
export const generationKinds: Record<string, string> = {
  full: '保存済みの設定で動画を生成', rerender: '動画のレンダリングを再実行',
  block_visual: '指定ブロックの画像を再生成', block_audio: '指定ブロックの音声を再生成',
};

export function settingChanges(response: LanguageResponse, history?: ProjectHistory) {
  const result = response.result;
  if (!result?.changed) return [];
  const before = history?.settings_versions.find((version) => version.revision === result.base_revision)?.settings;
  let after: Record<string, unknown> = {};
  let fields: string[] = [];
  if (['project.subtitle-font-size.set', 'project.subtitle-font-size.adjust'].includes(result.operation_id)) {
    after = { subtitle_font_size: result.data.subtitle_font_size };
    fields = ['subtitle_font_size'];
  } else if (['project.settings.update', 'project.settings.restore'].includes(result.operation_id)) {
    after = (result.data.settings ?? {}) as Record<string, unknown>;
    fields = Array.isArray(result.data.changed_fields) ? result.data.changed_fields.filter((field): field is string => typeof field === 'string') : [];
  }
  return fields.map((field) => ({ field, before: before?.[field], after: after[field] }));
}

const reasons: Record<string, string> = {
  stale_state: '設定が変更されています。最新の状態を確認して、もう一度依頼してください。',
  project_busy: '動画の生成中は設定変更や新しい生成を受け付けられません。',
  external_outcome_unknown: '外部処理の結果が未確定です。生成の履歴で復旧情報を確認してください。',
  model_not_configured: '自然言語用のモデルが未設定です。接続設定を確認してください。通常の設定フォームは利用できます。',
};

export function failureMessage(response: LanguageResponse): string | undefined {
  return response.failure ? reasons[response.failure.reason_code] ?? response.failure.message : undefined;
}
