/** TypeScript mirrors of the backend response and request contracts. */
/** Overall project lifecycle values stored by the backend. */
export type ProjectStatus =
  | 'pending'
  | 'splitting'
  | 'planning'
  | 'generating'
  | 'rendering'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'unknown';

/** Per-stage block status values. */
export type BlockStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface PronunciationOverride {
  surface: string;
  reading: string;
  accent: number | null;
}

export interface OutputQualitySettings {
  visual_focus_enabled: boolean;
  subtitle_mode: 'sentence' | 'packed';
  narration_pacing_mode: 'adaptive' | 'fixed';
  pronunciation_overrides: PronunciationOverride[];
}

/** Visual renderer values returned by the backend planner. */
export type VisualType =
  | 'ai_image'
  | 'code_slide'
  | 'diagram'
  | 'formula'
  | 'comparison'
  | 'title_slide'
  | 'text_slide';

/** Compact project row used by the project list. */
export interface ProjectSummary {
  id: number;
  revision: number;
  title: string;
  status: ProjectStatus;
  progress: number;
  current_stage: string | null;
  block_count: number;
  output_video_path: string | null;
  error_message: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export type ProjectGenerationRecoveryCode = 'busy' | 'external_outcome_unknown' | 'ready';
export type ProjectGenerationRecommendedAction = 'wait' | 'check_provider' | 'generate';

export interface ProjectGenerationRecovery {
  code: ProjectGenerationRecoveryCode;
  recommended_action: ProjectGenerationRecommendedAction;
}

/** Full project state returned by the detail and quick-create endpoints. */
export interface ProjectDetail extends ProjectSummary, OutputQualitySettings {
  generation_recovery: ProjectGenerationRecovery;
  source_script: string;
  global_visual_style: string | null;
  voicevox_url: string;
  voicevox_speaker_id: number;
  voicevox_speed_scale: number;
  voicevox_pitch_scale: number;
  voicevox_intonation_scale: number;
  voicevox_volume_scale: number;
  subtitle_enabled: boolean;
  subtitle_font_size: number;
  subtitle_position: string;
  subtitle_text_color: string;
  subtitle_outline_color: string;
  subtitle_background: boolean;
  subtitle_max_chars_per_line: number;
  pre_margin_seconds: number;
  post_margin_seconds: number;
  min_display_seconds: number;
  narration_sentence_pause_seconds: number;
  max_slides_per_block: number;
  use_fake_providers: boolean;
  output_subtitle_path: string | null;
}

/** One block's text, visual plan, stage statuses, and artifact URLs. */
export interface BlockSummary {
  id: number;
  project_id: number;
  index: number;
  source_text: string;
  tts_text: string;
  visual_type: VisualType;
  visual_plan: Record<string, unknown> | null;
  image_prompt: string | null;
  image_url: string | null;
  audio_url: string | null;
  video_url: string | null;
  duration_ms: number | null;
  display_duration_ms: number | null;
  status_split: BlockStatus;
  status_visual_plan: BlockStatus;
  status_image: BlockStatus;
  status_audio: BlockStatus;
  status_render: BlockStatus;
  error_message: string | null;
}

export type RecoveryCode =
  | 'wait'
  | 'safe_retry'
  | 'external_outcome_unknown'
  | 'refresh_required'
  | 'cancelled'
  | 'completed'
  | 'failed';

export type RecommendedAction = 'wait' | 'retry_current' | 'check_provider' | 'refresh' | 'none';

export interface StartupState {
  status: 'starting' | 'ready' | 'migration_failed';
  reason_code: string | null;
  message: string;
  schema_version: number | null;
  backup_available: boolean;
}

/** Progress and error state for an asynchronous generation job. */
export interface JobSummary {
  id: number;
  project_id: number;
  current_stage: string;
  status: string;
  progress: number;
  stage_progress: number;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  cancel_requested?: boolean;
  input_revision?: number | null;
  parent_job_id?: number | null;
  recovery_message?: string | null;
  retryable?: boolean;
  retry_blocked_reason?: string | null;
  recovery_code: RecoveryCode;
  recommended_action: RecommendedAction;
  plan?: { stages: string[] } | null;
}

export type ProjectSettings = Omit<CreateProjectInput, 'source_script' | 'use_fake_providers' | 'providers'>;

export interface VideoArtifact {
  id: number;
  job_id: number | null;
  revision: number | null;
  created_at: string;
  video_url: string;
  subtitle_url: string | null;
  is_current: boolean;
  available: boolean;
}

export interface SettingsVersion {
  revision: number;
  created_at: string;
  restored_from_revision: number | null;
  changed_fields: string[];
  settings: Record<string, unknown>;
}

export interface ProjectHistory {
  revision: number;
  output_state: 'none' | 'current' | 'stale' | 'missing';
  current_artifact_id: number | null;
  artifacts: VideoArtifact[];
  settings_versions: SettingsVersion[];
  jobs: JobSummary[];
}

export type ProjectOperationId =
  | 'project.status.get'
  | 'project.subtitle-font-size.set'
  | 'project.subtitle-font-size.adjust'
  | 'project.settings.update'
  | 'project.settings.restore'
  | 'project.artifact.restore'
  | 'project.generation.start'
  | 'project.generation.cancel'
  | 'project.generation.retry';

export type PartialGenerationKind = 'rerender' | 'block_visual' | 'block_audio';

export interface OperationRequest {
  request_id: string;
  operation_id: ProjectOperationId;
  operation_version: 1 | 2;
  target: { project_id: number };
  base_revision: number;
  arguments: Record<string, unknown>;
  generation_requested?: boolean;
}

export interface OperationResult {
  operation_id: string;
  request_id: string;
  revision: number;
  job_id: number | null;
  data: Record<string, unknown>;
}

/** Full form payload for the detailed project-creation screen. */
export interface CreateProjectInput extends OutputQualitySettings {
  title: string;
  source_script: string;
  voicevox_url: string;
  voicevox_speaker_id: number;
  voicevox_speed_scale: number;
  voicevox_pitch_scale: number;
  voicevox_intonation_scale: number;
  voicevox_volume_scale: number;
  subtitle_enabled: boolean;
  subtitle_font_size: number;
  subtitle_position: string;
  subtitle_text_color: string;
  subtitle_outline_color: string;
  subtitle_background: boolean;
  subtitle_max_chars_per_line: number;
  pre_margin_seconds: number;
  post_margin_seconds: number;
  min_display_seconds: number;
  narration_sentence_pause_seconds: number;
  max_slides_per_block: number;
  use_fake_providers: boolean;
  providers: {
    llm_api_key?: string;
    llm_base_url?: string;
    llm_model?: string;
    image_api_key?: string;
    image_base_url?: string;
    image_model?: string;
  };
}

/** One normalized VOICEVOX speaker and its styles. */
export interface SpeakerInfo {
  speaker_id: number;
  name: string;
  styles: Array<{ id: number; name?: string }>;
}

/** Speaker discovery response returned by the backend. */
export interface SpeakersEnvelope {
  url: string;
  speakers: SpeakerInfo[];
}
/** Minimal quick-create payload; omitted pacing values use server defaults. */
export interface QuickCreateInput extends Partial<OutputQualitySettings> {
  source_script: string;
  title?: string;
  voicevox_url?: string;
  voicevox_speaker_id?: number;
  use_fake_providers?: boolean;
  /** Pacing overrides. Omitted fields keep the project default. */
  voicevox_speed_scale?: number;
  narration_sentence_pause_seconds?: number;
  post_margin_seconds?: number;
  subtitle_font_size?: number;
  max_slides_per_block?: number;
}

export interface QuickCreateResponse {
  /** Project created by the quick endpoint. */
  project: ProjectDetail;
  /** Job immediately queued for that project. */
  job: JobSummary;
  message: string;
}
