import { describe, expect, it } from 'vitest';
import { createProjectSchema } from '@/lib/validation';

const VALID_DEFAULTS = {
  title: 'テスト動画',
  source_script: 'これは十分な長さを持つテスト用台本です。句点で区切ります。もう一つ。',
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
} as const;

describe('createProjectSchema', () => {
  it('rejects short scripts', () => {
    const result = createProjectSchema.safeParse({
      ...VALID_DEFAULTS,
      title: 't',
      source_script: '短い',
    });
    expect(result.success).toBe(false);
  });

  it('rejects invalid voicevox url', () => {
    const result = createProjectSchema.safeParse({
      ...VALID_DEFAULTS,
      voicevox_url: 'not-a-url',
    });
    expect(result.success).toBe(false);
  });

  it('rejects speed scale out of range', () => {
    const result = createProjectSchema.safeParse({
      ...VALID_DEFAULTS,
      voicevox_speed_scale: 3.0,
    });
    expect(result.success).toBe(false);
  });

  it('accepts a minimal valid input', () => {
    const result = createProjectSchema.safeParse(VALID_DEFAULTS);
    expect(result.success).toBe(true);
  });

  it('coerces number-like strings from form fields', () => {
    const result = createProjectSchema.safeParse({
      ...VALID_DEFAULTS,
      voicevox_speaker_id: '3' as unknown as number,
      voicevox_speed_scale: '1.2' as unknown as number,
    });
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.voicevox_speaker_id).toBe(3);
      expect(result.data.voicevox_speed_scale).toBeCloseTo(1.2);
    }
  });
});