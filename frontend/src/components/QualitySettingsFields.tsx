/** Shared output-quality controls for quick and detailed creation. */
import { useId } from 'react';
import type { OutputQualitySettings, PronunciationOverride } from '@/lib/types';
import { pronunciationOverridesSchema } from '@/lib/validation';

interface Props {
  value: OutputQualitySettings;
  onChange: (value: OutputQualitySettings) => void;
  disabled?: boolean;
}

export function QualitySettingsFields({ value, onChange, disabled }: Props) {
  const prefix = useId();
  const set = <K extends keyof OutputQualitySettings>(key: K, next: OutputQualitySettings[K]) =>
    onChange({ ...value, [key]: next });
  const updateReading = (index: number, patch: Partial<PronunciationOverride>) =>
    set('pronunciation_overrides', value.pronunciation_overrides.map((entry, i) =>
      i === index ? { ...entry, ...patch } : entry));
  const validation = pronunciationOverridesSchema.safeParse(value.pronunciation_overrides);

  return (
    <fieldset className="space-y-4" disabled={disabled}>
      <legend className="mb-3 text-sm font-semibold text-slate-800">見せ方・読み上げ</legend>
      <label className="flex items-start gap-2 text-sm text-slate-700">
        <input type="checkbox" className="mt-1" checked={value.visual_focus_enabled}
          onChange={(event) => set('visual_focus_enabled', event.target.checked)} />
        <span>説明中の図を強調する
          <span className="mt-1 block text-xs text-slate-500">1ブロックで1枚のスライドを表示するとき、対応する図では読み上げに合わせて注目箇所を切り替えます。</span>
        </span>
      </label>
      <div>
        <label className="label" htmlFor={`${prefix}-subtitle`}>字幕の切り替え</label>
        <select id={`${prefix}-subtitle`} className="input" value={value.subtitle_mode}
          onChange={(event) => set('subtitle_mode', event.target.value as OutputQualitySettings['subtitle_mode'])}>
          <option value="sentence">話している文ごと（おすすめ）</option>
          <option value="packed">複数の文をまとめる</option>
        </select>
      </div>
      <div>
        <label className="label" htmlFor={`${prefix}-pacing`}>読み上げの間</label>
        <select id={`${prefix}-pacing`} className="input" value={value.narration_pacing_mode}
          onChange={(event) => set('narration_pacing_mode', event.target.value as OutputQualitySettings['narration_pacing_mode'])}>
          <option value="adaptive">内容に合わせる（おすすめ）</option>
          <option value="fixed">文末で一定の間を置く</option>
        </select>
        <p className="mt-1 text-xs text-slate-500">
          通常の説明は0.6秒、図への注目や切り替えは1.0秒、長い説明や要点は1.5秒を目安に間を置きます。
        </p>
      </div>
      <fieldset className="space-y-3 rounded-md border border-slate-200 p-3">
        <legend className="px-1 text-sm font-medium text-slate-700">専門用語の読み方</legend>
        <p className="text-xs text-slate-500">
          字幕の表記はそのまま、読み方だけを指定します。アクセントは空欄で自動、0で平板、1以上で音が下がる位置を指定できます。
        </p>
        {value.pronunciation_overrides.map((entry, index) => (
          <div key={index} className="grid items-end gap-2 sm:grid-cols-[1fr_1fr_7rem_auto]">
            <label className="text-xs text-slate-600">
              表記 {index + 1}
              <input className="input mt-1" value={entry.surface} maxLength={80} placeholder="API"
                onChange={(event) => updateReading(index, { surface: event.target.value })} />
            </label>
            <label className="text-xs text-slate-600">
              読み方（カタカナ）{index + 1}
              <input className="input mt-1" value={entry.reading} maxLength={160} placeholder="エーピーアイ"
                onChange={(event) => updateReading(index, { reading: event.target.value })} />
            </label>
            <label className="text-xs text-slate-600">
              アクセント {index + 1}
              <input className="input mt-1" type="number" min={0} step={1} value={entry.accent ?? ''} placeholder="自動"
                onChange={(event) => updateReading(index, { accent: event.target.value === '' ? null : Number(event.target.value) })} />
            </label>
            <button type="button" className="btn-secondary" aria-label={`読み方 ${index + 1} を削除`}
              onClick={() => set('pronunciation_overrides', value.pronunciation_overrides.filter((_, i) => i !== index))}>
              削除
            </button>
          </div>
        ))}
        {!validation.success && (
          <ul className="space-y-1 text-xs text-red-600" role="alert">
            {validation.error.issues.map((issue, index) => (
              <li key={index}>{typeof issue.path[0] === 'number' ? `${issue.path[0] + 1}件目: ` : ''}{issue.message}</li>
            ))}
          </ul>
        )}
        <button type="button" className="btn-secondary text-xs" disabled={disabled || value.pronunciation_overrides.length >= 100}
          onClick={() => set('pronunciation_overrides', [...value.pronunciation_overrides, { surface: '', reading: '', accent: null }])}>
          読み方を追加
        </button>
      </fieldset>
    </fieldset>
  );
}
