import type { BlockSummary, JobSummary, ProjectDetail, ProjectHistory, RecoveryCode, RecommendedAction } from '@/lib/types';
import { DEFAULT_QUALITY } from '@/lib/validation';

export function projectFixture(overrides: Partial<ProjectDetail> = {}): ProjectDetail {
  return {
    id: 4, revision: 3, title: '説明動画', status: 'completed', progress: 1, current_stage: null,
    block_count: 1, output_video_path: null, output_subtitle_path: null, error_message: null,
    created_at: '2026-09-19T00:00:00Z', updated_at: '2026-09-19T00:00:00Z',
    source_script: 'この台本はテスト用の説明文章です。実際の外部サービスは呼び出しません。', global_visual_style: null,
    voicevox_url: 'http://127.0.0.1:50021', voicevox_speaker_id: 1, voicevox_speed_scale: 1,
    voicevox_pitch_scale: 0, voicevox_intonation_scale: 1, voicevox_volume_scale: 1,
    subtitle_enabled: true, subtitle_font_size: 48, subtitle_position: 'bottom',
    subtitle_text_color: '#FFFFFF', subtitle_outline_color: '#000000', subtitle_background: true,
    subtitle_max_chars_per_line: 36, pre_margin_seconds: 0.15, post_margin_seconds: 1.5,
    min_display_seconds: 2, narration_sentence_pause_seconds: 1.5, max_slides_per_block: 1,
    use_fake_providers: true,
    generation_recovery: { code: 'ready', recommended_action: 'generate' },
    ...DEFAULT_QUALITY, ...overrides,
  };
}

export function historyFixture(overrides: Partial<ProjectHistory> = {}): ProjectHistory {
  return {
    revision: 3, output_state: 'none', current_artifact_id: null, artifacts: [], jobs: [],
    settings_versions: [
      { revision: 3, created_at: '2026-09-19T00:03:00Z', restored_from_revision: null, changed_fields: ['subtitle_font_size'], settings: { subtitle_font_size: 48 } },
      { revision: 2, created_at: '2026-09-19T00:02:00Z', restored_from_revision: null, changed_fields: ['subtitle_font_size'], settings: { subtitle_font_size: 44 } },
      { revision: 1, created_at: '2026-09-19T00:01:00Z', restored_from_revision: null, changed_fields: [], settings: { subtitle_font_size: 40 } },
    ], ...overrides,
  };
}

export function jobFixture(overrides: Partial<JobSummary> = {}): JobSummary {
  const status = overrides.status ?? 'failed';
  const recoveryByStatus: Record<string, [RecoveryCode, RecommendedAction, boolean]> = {
    pending: ['wait', 'wait', false],
    running: ['wait', 'wait', false],
    unknown: ['external_outcome_unknown', 'check_provider', false],
    completed: ['completed', 'none', false],
    cancelled: ['safe_retry', 'retry_current', true],
    failed: ['safe_retry', 'retry_current', true],
  };
  const [recovery_code, recommended_action, retryable] = recoveryByStatus[status] ?? ['failed', 'none', false];
  return {
    id: 8, project_id: 4, current_stage: 'audio', status, progress: 0.5, stage_progress: 0,
    started_at: '2026-09-19T00:03:00Z', finished_at: null, error_message: null,
    cancel_requested: false, input_revision: 2, parent_job_id: null, recovery_message: null,
    retryable, retry_blocked_reason: null, recovery_code, recommended_action,
    plan: { stages: ['audio', 'render'] }, ...overrides,
  };
}

export function blockFixture(overrides: Partial<BlockSummary> = {}): BlockSummary {
  return {
    id: 52, project_id: 4, index: 6, source_text: '説明のテスト', tts_text: '説明のテスト',
    visual_type: 'text_slide', visual_plan: null, image_prompt: null,
    image_url: null, audio_url: null, video_url: null, duration_ms: null, display_duration_ms: null,
    status_split: 'completed', status_visual_plan: 'completed', status_image: 'completed',
    status_audio: 'completed', status_render: 'completed', error_message: null, ...overrides,
  };
}
