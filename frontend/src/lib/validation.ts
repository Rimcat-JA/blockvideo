/** Zod validation shared by the detailed project form and its submit handler. */
import { z } from 'zod';

/** Client-side constraints that mirror the backend ProjectCreate schema. */
export const createProjectSchema = z.object({
  title: z.string().min(1, 'タイトルを入力してください').max(255),
  source_script: z
    .string()
    .min(20, '台本は20文字以上で入力してください')
    .max(100_000, '台本は100000文字以内で入力してください'),
  voicevox_url: z.string().url('有効なURLを入力してください'),
  voicevox_speaker_id: z.coerce.number().int().nonnegative(),
  voicevox_speed_scale: z.coerce.number().min(0.5).max(2.0),
  voicevox_pitch_scale: z.coerce.number().min(-1).max(1),
  voicevox_intonation_scale: z.coerce.number().min(0).max(2),
  voicevox_volume_scale: z.coerce.number().min(0).max(2),
  subtitle_enabled: z.boolean(),
  subtitle_font_size: z.coerce.number().int().min(16).max(120),
  subtitle_position: z.enum(['top', 'middle', 'bottom']),
  subtitle_text_color: z.string(),
  subtitle_outline_color: z.string(),
  subtitle_background: z.boolean(),
  subtitle_max_chars_per_line: z.coerce.number().int().min(8).max(120),
  pre_margin_seconds: z.coerce.number().min(0).max(5),
  post_margin_seconds: z.coerce.number().min(0).max(5),
  min_display_seconds: z.coerce.number().min(0.5).max(10),
  use_fake_providers: z.boolean(),
});

export type CreateProjectForm = z.input<typeof createProjectSchema>;
export type CreateProjectInput = z.output<typeof createProjectSchema>;
