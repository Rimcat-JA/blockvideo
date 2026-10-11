/** Japanese labels for pipeline stage and status codes shown in the UI. */

/** Pipeline stages reported in ``current_stage``. */
export const STAGE_LABELS: Record<string, string> = {
  queued: '待機中',
  split: '台本の分割',
  plan: 'スライドの設計',
  image: 'スライドの描画',
  audio: '音声の合成',
  render: '動画の書き出し',
  concat: '動画の連結',
  subtitles: '字幕の作成',
  done: '完了',
  pending: '再開待ち',
  completed: '完了',
  failed: '失敗',
  cancelled: 'キャンセル済み',
  unknown: '結果未確定',
};

/** Project, job and block status values. */
export const STATUS_LABELS: Record<string, string> = {
  pending: '待機中',
  running: '実行中',
  splitting: '分割中',
  planning: '設計中',
  generating: '生成中',
  rendering: '書き出し中',
  completed: '完了',
  failed: '失敗',
  cancelled: 'キャンセル済み',
  skipped: 'スキップ',
  unknown: '結果未確定',
};

/** Label for a code, falling back to the code itself for unknown values. */
export function label(table: Record<string, string>, code: string): string {
  return table[code] ?? code;
}
