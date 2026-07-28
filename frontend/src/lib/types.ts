export type ProjectStatus =
  | 'pending'
  | 'splitting'
  | 'planning'
  | 'generating'
  | 'rendering'
  | 'completed'
  | 'failed'
  | 'cancelled';

export type BlockStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export type VisualType =
  | 'ai_image'
  | 'code_slide'
  | 'diagram'
  | 'formula'
  | 'comparison'
  | 'title_slide'
  | 'text_slide';

export interface ProjectSummary {
  id: number;
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

export interface ProjectDetail extends ProjectSummary {
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
  use_fake_providers: boolean;
  output_subtitle_path: string | null;
}

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
}

export interface CreateProjectInput {
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

export interface SpeakerInfo {
  speaker_id: number;
  name: string;
  styles: Array<{ id: number; name?: string }>;
}

export interface SpeakersEnvelope {
  url: string;
  speakers: SpeakerInfo[];
}
export interface QuickCreateInput {
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
  project: ProjectDetail;
  job: JobSummary;
  message: string;
}
