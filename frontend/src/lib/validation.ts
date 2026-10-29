/** Zod validation shared by the detailed project form and its submit handler. */
import { z } from 'zod';

export const pronunciationOverridesSchema = z.array(z.object({
  surface: z.string().trim().min(1, '表記を入力してください').max(80)
    .refine((text) => !/[。！？\n\r]/.test(text), '表記に文の区切りは使えません'),
  reading: z.string().trim().min(1, '読み方を入力してください').max(160)
    .regex(/^[ァ-ヴー]+$/, '読み方は全角カタカナで入力してください')
    .refine((text) => !/^[ァィゥェォャュョヮー]/.test(text), '読み方の先頭に小書き文字や長音は使えません'),
  accent: z.number().int().nonnegative().nullable(),
}).superRefine((entry, context) => {
  const moraCount = [...entry.reading].filter((char) => !'ァィゥェォャュョヮ'.includes(char)).length;
  if (entry.accent != null && entry.accent > moraCount) {
    context.addIssue({ code: 'custom', path: ['accent'], message: `アクセントは0〜${moraCount}で指定してください` });
  }
})).max(100, '読み方は100件まで登録できます').superRefine((entries, context) => {
  const surfaces = new Set<string>();
  entries.forEach((entry, index) => {
    if (surfaces.has(entry.surface)) {
      context.addIssue({ code: 'custom', path: [index, 'surface'], message: '同じ表記は1件だけ登録してください' });
    }
    surfaces.add(entry.surface);
  });
});

export const qualitySettingsSchema = z.object({
  visual_focus_enabled: z.boolean().default(true),
  subtitle_mode: z.enum(['sentence', 'packed']).default('sentence'),
  narration_pacing_mode: z.enum(['adaptive', 'fixed']).default('adaptive'),
  pronunciation_overrides: pronunciationOverridesSchema.default([]),
});

export const DEFAULT_QUALITY = qualitySettingsSchema.parse({});

/** A number field with Japanese messages; an emptied number input arrives as NaN. */
function numberIn(min: number, max: number, integer = false) {
  const base = z.coerce.number({ invalid_type_error: '数値を入力してください' });
  return (integer ? base.int('整数で入力してください') : base)
    .min(min, `${min}以上で入力してください`)
    .max(max, `${max}以下で入力してください`);
}

/** Client-side constraints that mirror the backend ProjectCreate schema. */
export const createProjectSchema = qualitySettingsSchema.extend({
  title: z.string().min(1, 'タイトルを入力してください').max(255),
  source_script: z
    .string()
    .min(20, '台本は20文字以上で入力してください')
    .max(100_000, '台本は100000文字以内で入力してください'),
  // Blank means the server's VOICEVOX_URL; anything typed must be an http(s) URL.
  voicevox_url: z.union([z.literal(''), z.string().regex(/^https?:\/\/[^\s/]+/, 'http:// か https:// で始まるURLを入力してください')]),
  voicevox_speaker_id: numberIn(0, 2 ** 31 - 1, true),
  voicevox_speed_scale: numberIn(0.5, 2.0),
  voicevox_pitch_scale: numberIn(-1, 1),
  voicevox_intonation_scale: numberIn(0, 2),
  voicevox_volume_scale: numberIn(0, 2),
  subtitle_enabled: z.boolean(),
  subtitle_font_size: numberIn(16, 120, true),
  subtitle_position: z.enum(['top', 'middle', 'bottom']),
  subtitle_text_color: z.string(),
  subtitle_outline_color: z.string(),
  subtitle_background: z.boolean(),
  subtitle_max_chars_per_line: numberIn(8, 120, true),
  pre_margin_seconds: numberIn(0, 5),
  post_margin_seconds: numberIn(0, 5),
  min_display_seconds: numberIn(0.5, 10),
  narration_sentence_pause_seconds: numberIn(0, 5).default(1.5),
  max_slides_per_block: numberIn(1, 9, true).default(1),
  use_fake_providers: z.boolean(),
});

export type CreateProjectForm = z.input<typeof createProjectSchema>;
export type CreateProjectInput = z.output<typeof createProjectSchema>;
