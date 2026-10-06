/** Edit saved settings separately from generation, with a revision-bound snapshot. */
import { useState } from 'react';
import type { ProjectDetail, ProjectSettings } from '@/lib/types';
import { createProjectSchema } from '@/lib/validation';
import { QualitySettingsFields } from '@/components/QualitySettingsFields';

const settingsSchema = createProjectSchema.omit({ source_script: true, use_fake_providers: true });

export function ProjectSettingsEditor({ project, disabled, onSave }: {
  project: ProjectDetail;
  disabled: boolean;
  onSave: (changes: Record<string, unknown>) => void;
}) {
  const [value, setValue] = useState<ProjectSettings>(() => Object.fromEntries(
    Object.keys(settingsSchema.shape).map((field) => [field, project[field as keyof ProjectSettings]]),
  ) as ProjectSettings);
  const validation = settingsSchema.safeParse(value);
  const changed = Object.entries(value).filter(([field, next]) =>
    JSON.stringify(next) !== JSON.stringify(project[field as keyof ProjectSettings]));
  const set = <K extends keyof ProjectSettings>(field: K, next: ProjectSettings[K]) =>
    setValue((previous) => ({ ...previous, [field]: next }));

  return (
    <section className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-semibold text-slate-800">現在の設定（版 {project.revision}）</h2>
      <p className="mt-1 text-sm text-slate-500">保存すると設定の履歴が残ります。動画へ反映するには、保存後に生成を開始してください。</p>
      {changed.length > 0 && <p className="mt-2 text-sm text-amber-700" role="status">未保存の変更があります。生成に使うには、先に「設定を保存」を押してください。</p>}
      <form className="mt-4" onSubmit={(event) => {
        event.preventDefault();
        if (disabled || !validation.success || !changed.length) return;
        onSave(Object.fromEntries(changed));
      }}>
        <fieldset disabled={disabled} className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-3">
            <label className="label">字幕の大きさ（px）
              <input className="input mt-1" type="number" min={16} max={120} step={1}
                value={value.subtitle_font_size} onChange={(event) => set('subtitle_font_size', Number(event.target.value))} />
            </label>
            <label className="label">話者ID
              <input className="input mt-1" type="number" min={0} step={1} value={value.voicevox_speaker_id}
                onChange={(event) => set('voicevox_speaker_id', Number(event.target.value))} />
            </label>
            <label className="label">話す速さ
              <input className="input mt-1" type="number" min={0.5} max={2} step={0.05} value={value.voicevox_speed_scale}
                onChange={(event) => set('voicevox_speed_scale', Number(event.target.value))} />
            </label>
          </div>
          <QualitySettingsFields value={value} onChange={(quality) => setValue({ ...value, ...quality })} disabled={disabled} />
          <details className="rounded border border-slate-200 p-3">
            <summary className="cursor-pointer text-sm font-medium text-slate-700">その他の設定</summary>
            <div className="mt-3 grid gap-4 sm:grid-cols-2">
              <label className="label">動画タイトル
                <input className="input mt-1" value={value.title} onChange={(event) => set('title', event.target.value)} />
              </label>
              <label className="label">VOICEVOX Engine URL
                <input className="input mt-1" type="url" value={value.voicevox_url} onChange={(event) => set('voicevox_url', event.target.value)} />
              </label>
              {([
                ['voicevox_pitch_scale', '声の高さ', -1, 1, 0.05],
                ['voicevox_intonation_scale', '抑揚', 0, 2, 0.1],
                ['voicevox_volume_scale', '音量', 0, 2, 0.1],
                ['subtitle_max_chars_per_line', '字幕1行の最大文字数', 8, 120, 1],
                ['pre_margin_seconds', '前余白（秒）', 0, 5, 0.05],
                ['post_margin_seconds', '後余白（秒）', 0, 5, 0.05],
                ['min_display_seconds', '最低表示時間（秒）', 0.5, 10, 0.5],
                ['narration_sentence_pause_seconds', '一定の文末の間（秒）', 0, 5, 0.1],
                ['max_slides_per_block', '1ブロックの最大スライド枚数', 1, 9, 1],
              ] as const).map(([field, label, min, max, step]) => (
                <label key={field} className="label">{label}
                  <input className="input mt-1" type="number" min={min} max={max} step={step} value={value[field]}
                    onChange={(event) => set(field, Number(event.target.value))} />
                </label>
              ))}
              <label className="label">字幕の位置
                <select className="input mt-1" value={value.subtitle_position} onChange={(event) => set('subtitle_position', event.target.value)}>
                  <option value="top">上</option><option value="middle">中央</option><option value="bottom">下</option>
                </select>
              </label>
              {([['subtitle_text_color', '字幕の文字色'], ['subtitle_outline_color', '字幕の縁取り色']] as const).map(([field, label]) => (
                <label key={field} className="label">{label}
                  <input type="color" className="input mt-1 h-10 p-1" value={value[field]} onChange={(event) => set(field, event.target.value)} />
                </label>
              ))}
              {([['subtitle_enabled', '字幕を表示する'], ['subtitle_background', '字幕の半透明背景を表示する']] as const).map(([field, label]) => (
                <label key={field} className="flex items-center gap-2 text-sm text-slate-700">
                  <input type="checkbox" checked={value[field]} onChange={(event) => set(field, event.target.checked)} />{label}
                </label>
              ))}
            </div>
          </details>
          {!validation.success && <p role="alert" className="text-sm text-red-600">入力値を確認してください。{validation.error.issues[0]?.message}</p>}
          <button className="btn-primary" type="submit" disabled={disabled || !validation.success || !changed.length}>設定を保存</button>
        </fieldset>
      </form>
    </section>
  );
}
