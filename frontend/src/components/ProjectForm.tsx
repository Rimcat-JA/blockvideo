/** Detailed project-creation form for scripts, providers, VOICEVOX, and ASS. */
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useEffect, useState, type ReactNode } from 'react';
import { useCreateProject, useProjectDefaults, useSpeakers } from '@/api/hooks';
import { useNavigate } from 'react-router-dom';
import { createProjectSchema, DEFAULT_QUALITY, type CreateProjectForm, type CreateProjectInput } from '@/lib/validation';
import { QualitySettingsFields } from '@/components/QualitySettingsFields';

/** Fields the server fills from VOICEVOX_* / NARRATION_* when the request omits them. */
const SERVER_DEFAULTED = [
  'voicevox_speaker_id', 'voicevox_speed_scale', 'voicevox_pitch_scale', 'voicevox_intonation_scale',
  'voicevox_volume_scale', 'narration_sentence_pause_seconds',
] as const;

type NumberFieldName =
  | 'voicevox_speed_scale' | 'voicevox_pitch_scale' | 'voicevox_intonation_scale' | 'voicevox_volume_scale'
  | 'narration_sentence_pause_seconds' | 'max_slides_per_block' | 'subtitle_font_size'
  | 'subtitle_max_chars_per_line' | 'pre_margin_seconds' | 'post_margin_seconds' | 'min_display_seconds';

/** Wait until typing pauses, so the speaker list is not fetched for every partial URL. */
function useDebounced<T>(value: T, delayMs: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return settled;
}

export function ProjectForm() {
  /** Validate the form, submit provider settings, and navigate to the project. */
  const navigate = useNavigate();
  const create = useCreateProject();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting, dirtyFields },
    watch,
    setValue,
    getValues,
    reset,
  } = useForm<CreateProjectForm, unknown, CreateProjectInput>({
    resolver: zodResolver(createProjectSchema),
    defaultValues: {
      ...DEFAULT_QUALITY,
      title: '',
      source_script: '',
      voicevox_url: '',
      voicevox_speaker_id: 1,
      voicevox_speed_scale: 1.0,
      voicevox_pitch_scale: 0.0,
      voicevox_intonation_scale: 1.0,
      voicevox_volume_scale: 1.0,
      subtitle_enabled: true,
      subtitle_font_size: 48,
      subtitle_position: 'bottom',
      subtitle_text_color: '#FFFFFF',
      subtitle_outline_color: '#000000',
      subtitle_background: true,
      subtitle_max_chars_per_line: 36,
      pre_margin_seconds: 0.15,
      post_margin_seconds: 1.5,
      min_display_seconds: 2.0,
      narration_sentence_pause_seconds: 1.5,
      max_slides_per_block: 1,
      use_fake_providers: false,
    },
  });

  // Show the server's .env defaults, so an untouched field displays what the
  // project will get and a deliberately chosen equal value is still honored.
  const serverDefaults = useProjectDefaults().data;
  useEffect(() => {
    if (serverDefaults) reset({ ...getValues(), ...serverDefaults }, { keepDirtyValues: true, keepErrors: true });
  }, [serverDefaults, reset, getValues]);

  const useFake = watch('use_fake_providers');
  const voicevoxUrl = useDebounced(watch('voicevox_url'), 500);
  // List the speakers of the engine this project will use (blank: the server default).
  const speakers = useSpeakers(/^https?:\/\/[^\s/]+/.test(voicevoxUrl ?? '') ? voicevoxUrl : undefined);

  /** A number input with a linked label and its validation message. */
  const numberField = (
    name: NumberFieldName,
    label: ReactNode,
    attrs: { min: number; max: number; step?: number; disabled?: boolean },
    hint?: ReactNode,
  ) => (
    <div className="space-y-1">
      <label className="label" htmlFor={name}>{label}</label>
      <input id={name} type="number" className="input" {...attrs} {...register(name, { valueAsNumber: true })} />
      {hint}
      {errors[name] && <p className="text-xs text-red-600">{errors[name]?.message}</p>}
    </div>
  );

  const onSubmit = handleSubmit(async (form) => {
    const llmApiKey = (document.getElementById('llm_api_key') as HTMLInputElement | null)?.value ?? '';
    const llmBaseUrl = (document.getElementById('llm_base_url') as HTMLInputElement | null)?.value ?? '';
    const llmModel = (document.getElementById('llm_model') as HTMLInputElement | null)?.value ?? '';
    const imageApiKey = (document.getElementById('image_api_key') as HTMLInputElement | null)?.value ?? '';
    const imageModel = (document.getElementById('image_model') as HTMLInputElement | null)?.value ?? '';
    // Untouched VOICEVOX and pause fields defer to the server's .env defaults.
    const serverDefaulted = Object.fromEntries(SERVER_DEFAULTED
      .filter((field) => !dirtyFields[field]).map((field) => [field, undefined]));
    const result = await create.mutateAsync({
      ...form,
      ...serverDefaulted,
      voicevox_url: form.voicevox_url || undefined,
      providers: {
        llm_api_key: llmApiKey || undefined,
        llm_base_url: llmBaseUrl || undefined,
        llm_model: llmModel || undefined,
        image_api_key: imageApiKey || undefined,
        image_model: imageModel || undefined,
      },
    });
    navigate(`/projects/${result.id}`);
  });

  return (
    <form onSubmit={onSubmit} className="space-y-6" noValidate>
      <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-800">台本</h2>
        <div className="space-y-1">
          <label className="label" htmlFor="title">
            動画タイトル
          </label>
          <input id="title" className="input" placeholder="例: Compose Multiplatform 入門" {...register('title')} />
          {errors.title && <p className="text-xs text-red-600">{errors.title.message}</p>}
        </div>
        <div className="space-y-1">
          <label className="label" htmlFor="source_script">
            完成済みの日本語台本
          </label>
          <textarea
            id="source_script"
            rows={12}
            className="input font-mono text-sm"
            placeholder="ここに完成済みの台本を貼り付けてください"
            {...register('source_script')}
          />
          {errors.source_script && (
            <p className="text-xs text-red-600">{errors.source_script.message}</p>
          )}
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" {...register('use_fake_providers')} />
          <span>デモ用: 外部APIを使わずFakeProviderで実行する(APIキー不要)</span>
        </label>
      </section>

      <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-800">LLM / 画像生成API キー</h2>
          <button
            type="button"
            className="text-xs text-accent-700 hover:underline"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '折りたたむ' : '詳細設定'}
          </button>
        </div>
        <p className="text-xs text-slate-500">
          {useFake
            ? 'デモモードが有効です。LLM/画像APIキーは無視されます。'
            : 'APIキーはメモリ上だけに保持され、ログやデータベースへ保存されません。'}
        </p>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-1">
            <label className="label" htmlFor="llm_api_key">
              LLM APIキー
            </label>
            <input id="llm_api_key" type="password" className="input" placeholder="sk-..." disabled={useFake} />
          </div>
          <div className="space-y-1">
            <label className="label" htmlFor="llm_base_url">
              LLM Base URL
            </label>
            <input id="llm_base_url" className="input" placeholder="https://api.openai.com/v1" disabled={useFake} />
          </div>
          <div className="space-y-1">
            <label className="label" htmlFor="llm_model">
              LLM モデル名
            </label>
            <input id="llm_model" className="input" placeholder="gpt-4o-mini" disabled={useFake} />
          </div>
          <div className="space-y-1">
            <label className="label" htmlFor="image_api_key">
              画像APIキー (任意)
            </label>
            <input id="image_api_key" type="password" className="input" placeholder="sk-..." disabled={useFake} />
          </div>
          <div className="space-y-1">
            <label className="label" htmlFor="image_model">
              画像モデル名
            </label>
            <input id="image_model" className="input" placeholder="gpt-image-1" disabled={useFake} />
          </div>
        </div>
        {showAdvanced && (
          <p className="text-xs text-slate-500">
            ※ 画像生成APIキーが未入力の場合、AI画像を要求するブロックは自動的にローカル描画スライドへ切り替わります。
          </p>
        )}
      </section>

      <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-800">VOICEVOX</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="space-y-1">
            <label className="label" htmlFor="voicevox_url">
              VOICEVOX Engine URL
            </label>
            <input
              id="voicevox_url"
              className="input"
              placeholder={speakers.data?.url ?? 'サーバーの既定（VOICEVOX_URL）'}
              {...register('voicevox_url')}
            />
            <p className="text-xs text-slate-500">空欄のときはサーバーの既定（.env の VOICEVOX_URL）を使います。</p>
            {errors.voicevox_url && <p className="text-xs text-red-600">{errors.voicevox_url.message}</p>}
          </div>
          <div className="space-y-1">
            <label className="label" htmlFor="voicevox_speaker_id">
              話者
            </label>
            <select
              id="voicevox_speaker_id"
              className="input"
              // Controlled: when the speaker list loads, the browser would otherwise
              // show its first option while the form still holds another id.
              value={String(watch('voicevox_speaker_id'))}
              onChange={(e) => setValue('voicevox_speaker_id', Number(e.target.value), { shouldDirty: true })}
            >
              {(speakers.data?.speakers ?? []).flatMap((s) =>
                (s.styles.length ? s.styles : [{ id: s.speaker_id, name: undefined }]).map((style) => (
                  <option key={`${s.speaker_id}-${style.id}`} value={style.id}>
                    {s.name}
                    {style.name ? `（${style.name}）` : ''}
                  </option>
                )),
              )}
              {!(speakers.data?.speakers ?? []).some((s) =>
                (s.styles.length ? s.styles.map((style) => style.id) : [s.speaker_id]).includes(watch('voicevox_speaker_id')),
              ) && (
                <option value={watch('voicevox_speaker_id')}>
                  話者ID {watch('voicevox_speaker_id')}{speakers.data == null ? '（VOICEVOX未接続）' : ''}
                </option>
              )}
            </select>
            {speakers.error && (
              <p className="text-xs text-amber-600">VOICEVOX Engineに接続できません。</p>
            )}
          </div>
          {numberField('voicevox_speed_scale', '話速 (speedScale)', { min: 0.5, max: 2, step: 0.1 })}
          {numberField('voicevox_pitch_scale', '音高 (pitchScale)', { min: -1, max: 1, step: 0.05 })}
          {numberField('voicevox_intonation_scale', '抑揚 (intonationScale)', { min: 0, max: 2, step: 0.1 })}
          {numberField('voicevox_volume_scale', '音量 (volumeScale)', { min: 0, max: 2, step: 0.1 })}
        </div>
      </section>

      <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <QualitySettingsFields
          value={{
            visual_focus_enabled: watch('visual_focus_enabled') ?? DEFAULT_QUALITY.visual_focus_enabled,
            subtitle_mode: watch('subtitle_mode') ?? DEFAULT_QUALITY.subtitle_mode,
            narration_pacing_mode: watch('narration_pacing_mode') ?? DEFAULT_QUALITY.narration_pacing_mode,
            pronunciation_overrides: watch('pronunciation_overrides') ?? [],
          }}
          onChange={(quality) => {
            setValue('visual_focus_enabled', quality.visual_focus_enabled, { shouldValidate: true });
            setValue('subtitle_mode', quality.subtitle_mode, { shouldValidate: true });
            setValue('narration_pacing_mode', quality.narration_pacing_mode, { shouldValidate: true });
            setValue('pronunciation_overrides', quality.pronunciation_overrides, { shouldValidate: true });
          }}
          disabled={isSubmitting || create.isPending}
        />
        <div className="grid gap-4 md:grid-cols-2">
          {numberField('narration_sentence_pause_seconds', '文末の息継ぎ（秒）',
            { min: 0, max: 5, step: 0.1, disabled: watch('narration_pacing_mode') !== 'fixed' },
            <span className="block text-xs text-slate-500">一定の間を置く場合に使用します。0で文間に間を入れません。</span>)}
          {numberField('max_slides_per_block', '1ブロックの最大スライド枚数', { min: 1, max: 9 })}
        </div>
      </section>

      <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-800">字幕設定</h2>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" {...register('subtitle_enabled')} />
          字幕を表示する
        </label>
        <div className="grid gap-4 md:grid-cols-3">
          {numberField('subtitle_font_size', 'フォントサイズ', { min: 16, max: 120 })}
          <div className="space-y-1">
            <label className="label" htmlFor="subtitle_position">位置</label>
            <select id="subtitle_position" className="input" {...register('subtitle_position')}>
              <option value="top">上</option>
              <option value="middle">中央</option>
              <option value="bottom">下</option>
            </select>
          </div>
          {numberField('subtitle_max_chars_per_line', '1行の最大文字数', { min: 8, max: 120 })}
          <div className="space-y-1">
            <label className="label" htmlFor="subtitle_text_color">文字色</label>
            <input id="subtitle_text_color" type="color" className="input h-10 p-1" {...register('subtitle_text_color')} />
          </div>
          <div className="space-y-1">
            <label className="label" htmlFor="subtitle_outline_color">縁取り色</label>
            <input id="subtitle_outline_color" type="color" className="input h-10 p-1" {...register('subtitle_outline_color')} />
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" {...register('subtitle_background')} />
            半透明背景を表示
          </label>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {numberField('pre_margin_seconds', '前余白 (秒)', { min: 0, max: 5, step: 0.05 })}
          {numberField('post_margin_seconds', '後余白 (秒)', { min: 0, max: 5, step: 0.05 })}
          {numberField('min_display_seconds', '最低表示時間 (秒)', { min: 0.5, max: 10, step: 0.5 })}
        </div>
      </section>

      {create.error && (
        <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {(create.error as Error).message}
        </div>
      )}

      <div className="flex items-center justify-end gap-3">
        <button
          type="submit"
          className="btn-primary"
          disabled={isSubmitting || create.isPending}
        >
          {create.isPending ? '作成中...' : 'プロジェクトを作成'}
        </button>
      </div>
    </form>
  );
}
