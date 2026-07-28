import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useState } from 'react';
import { useCreateProject, useSpeakers } from '@/api/hooks';
import { useNavigate } from 'react-router-dom';
import { createProjectSchema, type CreateProjectForm } from '@/lib/validation';

export function ProjectForm() {
  const navigate = useNavigate();
  const create = useCreateProject();
  const speakers = useSpeakers();
  const [showAdvanced, setShowAdvanced] = useState(false);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    watch,
    setValue,
  } = useForm<CreateProjectForm>({
    resolver: zodResolver(createProjectSchema),
    defaultValues: {
      title: '',
      source_script: '',
      voicevox_url: 'http://127.0.0.1:50021',
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
      post_margin_seconds: 0.35,
      min_display_seconds: 2.0,
      use_fake_providers: false,
    },
  });

  const useFake = watch('use_fake_providers');

  const onSubmit = handleSubmit(async (rawForm) => {
    const form = rawForm as unknown as import('@/lib/validation').CreateProjectInput;
    const llmApiKey = (document.getElementById('llm_api_key') as HTMLInputElement | null)?.value ?? '';
    const llmBaseUrl = (document.getElementById('llm_base_url') as HTMLInputElement | null)?.value ?? '';
    const llmModel = (document.getElementById('llm_model') as HTMLInputElement | null)?.value ?? '';
    const imageApiKey = (document.getElementById('image_api_key') as HTMLInputElement | null)?.value ?? '';
    const imageModel = (document.getElementById('image_model') as HTMLInputElement | null)?.value ?? '';
    const result = await create.mutateAsync({
      ...form,
      providers: {
        llm_api_key: llmApiKey || undefined,
        llm_base_url: llmBaseUrl || undefined,
        llm_model: llmModel || undefined,
        image_api_key: imageApiKey || undefined,
        image_model: imageModel || undefined,
        image_base_url: 'https://api.openai.com/v1',
      },
    });
    navigate(`/projects/${result.id}`);
  });

  return (
    <form onSubmit={onSubmit} className="space-y-6">
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
            <input id="voicevox_url" className="input" {...register('voicevox_url')} />
            {errors.voicevox_url && <p className="text-xs text-red-600">{errors.voicevox_url.message}</p>}
          </div>
          <div className="space-y-1">
            <label className="label" htmlFor="voicevox_speaker_id">
              話者
            </label>
            <select
              id="voicevox_speaker_id"
              className="input"
              {...register('voicevox_speaker_id', { valueAsNumber: true })}
              onChange={(e) => setValue('voicevox_speaker_id', Number(e.target.value))}
            >
              {(speakers.data?.speakers ?? []).map((s) => (
                <option key={s.speaker_id} value={s.speaker_id}>
                  {s.speaker_id}: {s.name}
                </option>
              ))}
              {speakers.data == null && <option value={1}>1 (VOICEVOX未接続)</option>}
            </select>
            {speakers.error && (
              <p className="text-xs text-amber-600">VOICEVOX Engineに接続できません。</p>
            )}
          </div>
          <div className="space-y-1">
            <label className="label">話速 (speedScale)</label>
            <input
              type="number"
              step={0.1}
              min={0.5}
              max={2}
              className="input"
              {...register('voicevox_speed_scale', { valueAsNumber: true })}
            />
          </div>
          <div className="space-y-1">
            <label className="label">音高 (pitchScale)</label>
            <input
              type="number"
              step={0.05}
              min={-1}
              max={1}
              className="input"
              {...register('voicevox_pitch_scale', { valueAsNumber: true })}
            />
          </div>
          <div className="space-y-1">
            <label className="label">抑揚 (intonationScale)</label>
            <input
              type="number"
              step={0.1}
              min={0}
              max={2}
              className="input"
              {...register('voicevox_intonation_scale', { valueAsNumber: true })}
            />
          </div>
          <div className="space-y-1">
            <label className="label">音量 (volumeScale)</label>
            <input
              type="number"
              step={0.1}
              min={0}
              max={2}
              className="input"
              {...register('voicevox_volume_scale', { valueAsNumber: true })}
            />
          </div>
        </div>
      </section>

      <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-slate-800">字幕設定</h2>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" {...register('subtitle_enabled')} />
          字幕を表示する
        </label>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-1">
            <label className="label">フォントサイズ</label>
            <input
              type="number"
              className="input"
              min={16}
              max={120}
              {...register('subtitle_font_size', { valueAsNumber: true })}
            />
          </div>
          <div className="space-y-1">
            <label className="label">位置</label>
            <select className="input" {...register('subtitle_position')}>
              <option value="top">上</option>
              <option value="middle">中央</option>
              <option value="bottom">下</option>
            </select>
          </div>
          <div className="space-y-1">
            <label className="label">1行の最大文字数</label>
            <input
              type="number"
              className="input"
              min={8}
              max={120}
              {...register('subtitle_max_chars_per_line', { valueAsNumber: true })}
            />
          </div>
          <div className="space-y-1">
            <label className="label">文字色</label>
            <input type="color" className="input h-10 p-1" {...register('subtitle_text_color')} />
          </div>
          <div className="space-y-1">
            <label className="label">縁取り色</label>
            <input type="color" className="input h-10 p-1" {...register('subtitle_outline_color')} />
          </div>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" {...register('subtitle_background')} />
            半透明背景を表示
          </label>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="space-y-1">
            <label className="label">前余白 (秒)</label>
            <input
              type="number"
              step={0.05}
              min={0}
              max={5}
              className="input"
              {...register('pre_margin_seconds', { valueAsNumber: true })}
            />
          </div>
          <div className="space-y-1">
            <label className="label">後余白 (秒)</label>
            <input
              type="number"
              step={0.05}
              min={0}
              max={5}
              className="input"
              {...register('post_margin_seconds', { valueAsNumber: true })}
            />
          </div>
          <div className="space-y-1">
            <label className="label">最低表示時間 (秒)</label>
            <input
              type="number"
              step={0.5}
              min={0.5}
              max={10}
              className="input"
              {...register('min_display_seconds', { valueAsNumber: true })}
            />
          </div>
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