/** Shared human-readable setting labels for recorded history and result cards. */
export const settingNames: Record<string, string> = {
  title: 'タイトル', subtitle_font_size: '字幕の大きさ', subtitle_mode: '字幕の切り替え',
  subtitle_enabled: '字幕を表示', subtitle_position: '字幕の位置', subtitle_text_color: '字幕の色',
  subtitle_outline_color: '字幕の縁取り色', subtitle_background: '字幕の背景', subtitle_max_chars_per_line: '字幕1行の最大文字数',
  voicevox_url: '音声エンジンURL', voicevox_speaker_id: '話者ID', voicevox_speed_scale: '話す速さ',
  voicevox_pitch_scale: '声の高さ', voicevox_intonation_scale: '抑揚', voicevox_volume_scale: '音量',
  visual_focus_enabled: '図の強調', pronunciation_overrides: '読み方の指定', narration_pacing_mode: '読み上げの間',
  narration_sentence_pause_seconds: '文末の間', max_slides_per_block: '最大スライド枚数',
  pre_margin_seconds: '前余白', post_margin_seconds: '後余白', min_display_seconds: '最低表示時間',
};
const values: Record<string, string> = { sentence: '文ごと', packed: '複数の文をまとめる', adaptive: '内容に合わせる', fixed: '一定', top: '上', middle: '中央', bottom: '下' };

export function displaySetting(value: unknown): string {
  if (typeof value === 'boolean') return value ? '有効' : '無効';
  if (typeof value === 'string') return values[value] ?? value;
  if (Array.isArray(value)) return value.length ? value.map((item) => {
    if (item && typeof item === 'object' && 'surface' in item && 'reading' in item)
      return `${String(item.surface)} → ${String(item.reading)}`;
    return String(item);
  }).join('、') : 'なし';
  return String(value ?? 'なし');
}

export function settingValue(field: string, value: unknown): string {
  return displaySetting(value) + (field === 'subtitle_font_size' && typeof value === 'number' ? 'px' : '');
}
