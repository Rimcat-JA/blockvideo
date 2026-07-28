/** Per-project narration, slide, subtitle, and speaker controls. */
import { useSpeakers } from '@/api/hooks';

/** The tempo knobs a viewer actually feels, and their defaults. */
export interface Pacing {
  narration_sentence_pause_seconds: number;
  voicevox_speed_scale: number;
  post_margin_seconds: number;
  max_slides_per_block: number;
  subtitle_font_size: number;
  voicevox_speaker_id: number;
}

export const DEFAULT_PACING: Pacing = {
  narration_sentence_pause_seconds: 1.5,
  voicevox_speed_scale: 1.0,
  post_margin_seconds: 1.5,
  max_slides_per_block: 1,
  subtitle_font_size: 48,
  voicevox_speaker_id: 1,
};

interface SliderProps {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  fallback: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}

function Slider({
  label, hint, value, min, max, step, unit, fallback, disabled, onChange,
}: SliderProps) {
  /** Render one numeric range control with default-value feedback. */
  const changed = Math.abs(value - fallback) > 1e-9;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <span className="text-sm tabular-nums text-slate-600">
          {step < 1 ? value.toFixed(step < 0.1 ? 2 : 1) : value}
          {unit}
          {changed && <span className="ml-2 text-xs text-accent-600">既定 {fallback}{unit}</span>}
        </span>
      </div>
      <input
        type="range"
        className="mt-1 w-full accent-accent-600 disabled:opacity-50"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <p className="mt-0.5 text-xs text-slate-500">{hint}</p>
    </div>
  );
}

interface Props {
  value: Pacing;
  onChange: (next: Pacing) => void;
  voicevoxUrl?: string;
  disabled?: boolean;
}

/**
 * Pacing controls for the quick-generate flow.
 *
 * These are per-project on the server, so changing one here affects only the
 * video about to be made — an existing project keeps whatever it was built
 * with.
 */
export function PacingSettings({ value, onChange, voicevoxUrl, disabled }: Props) {
  const speakers = useSpeakers(voicevoxUrl);
  const set = <K extends keyof Pacing>(key: K, next: Pacing[K]) =>
    onChange({ ...value, [key]: next });
  const isDefault = (Object.keys(DEFAULT_PACING) as Array<keyof Pacing>).every(
    (k) => Math.abs(value[k] - DEFAULT_PACING[k]) < 1e-9,
  );

  return (
    <details className="rounded-lg border border-slate-200 bg-white">
      <summary className="cursor-pointer select-none px-4 py-3 text-sm font-medium text-slate-700">
        カスタム設定
        <span className="ml-2 text-xs font-normal text-slate-500">
          {isDefault ? '（既定のまま）' : '（変更あり）'}
        </span>
      </summary>

      <div className="space-y-5 border-t border-slate-100 px-4 py-4">
        <Slider
          label="文末の息継ぎ"
          hint="「。」ごとに置く間。長いほど落ち着いた印象になります（0 で VOICEVOX 標準の約0.4秒）。"
          value={value.narration_sentence_pause_seconds}
          min={0} max={3} step={0.1} unit="秒"
          fallback={DEFAULT_PACING.narration_sentence_pause_seconds}
          disabled={disabled}
          onChange={(v) => set('narration_sentence_pause_seconds', v)}
        />
        <Slider
          label="読み上げ速度"
          hint="1.0 が標準。上げると全体が短くなります。"
          value={value.voicevox_speed_scale}
          min={0.5} max={2} step={0.05} unit="倍"
          fallback={DEFAULT_PACING.voicevox_speed_scale}
          disabled={disabled}
          onChange={(v) => set('voicevox_speed_scale', v)}
        />
        <Slider
          label="ブロック間の余韻"
          hint="読み終えてから次のスライドへ移るまで、図を見る時間。"
          value={value.post_margin_seconds}
          min={0} max={5} step={0.1} unit="秒"
          fallback={DEFAULT_PACING.post_margin_seconds}
          disabled={disabled}
          onChange={(v) => set('post_margin_seconds', v)}
        />
        <Slider
          label="1ブロックの最大スライド枚数"
          hint="1枚あたりの表示時間を確保するための上限。超えたぶんは作られません（1枚なら図だけを最後まで見せます）。"
          value={value.max_slides_per_block}
          min={1} max={8} step={1} unit="枚"
          fallback={DEFAULT_PACING.max_slides_per_block}
          disabled={disabled}
          onChange={(v) => set('max_slides_per_block', v)}
        />
        <Slider
          label="字幕の文字サイズ"
          hint="収まらない場合は自動で縮みます。これは上限です。"
          value={value.subtitle_font_size}
          min={24} max={72} step={2} unit="px"
          fallback={DEFAULT_PACING.subtitle_font_size}
          disabled={disabled}
          onChange={(v) => set('subtitle_font_size', v)}
        />

        <div>
          <label className="text-sm font-medium text-slate-700" htmlFor="speaker">
            話者
          </label>
          <select
            id="speaker"
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm
                       focus:border-accent-500 focus:outline-none focus:ring-1 focus:ring-accent-500
                       disabled:opacity-50"
            value={value.voicevox_speaker_id}
            disabled={disabled}
            onChange={(e) => set('voicevox_speaker_id', Number(e.target.value))}
          >
            {speakers.data?.speakers.flatMap((s) =>
              (s.styles.length ? s.styles : [{ id: s.speaker_id, name: undefined }]).map((style) => (
                <option key={`${s.speaker_id}-${style.id}`} value={style.id}>
                  {s.name}
                  {style.name ? `（${style.name}）` : ''}
                </option>
              )),
            ) ?? <option value={value.voicevox_speaker_id}>読み込み中…</option>}
          </select>
          {speakers.isError && (
            <p className="mt-1 text-xs text-amber-600">
              VOICEVOX の話者一覧を取得できませんでした。Engine の起動を確認してください。
            </p>
          )}
        </div>

        <button
          type="button"
          className="text-xs text-slate-500 underline disabled:opacity-50"
          disabled={disabled || isDefault}
          onClick={() => onChange(DEFAULT_PACING)}
        >
          既定値に戻す
        </button>
      </div>
    </details>
  );
}
