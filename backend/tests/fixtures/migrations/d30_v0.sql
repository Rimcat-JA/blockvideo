BEGIN TRANSACTION;
CREATE TABLE blocks (
    id INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    "index" INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    tts_text TEXT NOT NULL,
    visual_type VARCHAR(32) NOT NULL,
    visual_plan_json JSON,
    image_prompt TEXT,
    image_path VARCHAR(1024),
    audio_path VARCHAR(1024),
    video_path VARCHAR(1024),
    duration_ms INTEGER,
    display_duration_ms INTEGER,
    status_split VARCHAR(16) NOT NULL,
    status_visual_plan VARCHAR(16) NOT NULL,
    status_image VARCHAR(16) NOT NULL,
    status_audio VARCHAR(16) NOT NULL,
    status_render VARCHAR(16) NOT NULL,
    error_message TEXT,
    content_hash VARCHAR(64),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(project_id) REFERENCES projects (id) ON DELETE CASCADE
);
CREATE TABLE external_calls (
    id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    fingerprint VARCHAR(64) NOT NULL,
    provider VARCHAR(64) NOT NULL,
    endpoint VARCHAR(512) NOT NULL,
    remote_side_effect BOOLEAN NOT NULL,
    status VARCHAR(16) NOT NULL,
    attempts INTEGER NOT NULL,
    response_status INTEGER,
    response_body BLOB,
    response_content_type VARCHAR(128),
    provider_response_id VARCHAR(128),
    error_code VARCHAR(64),
    started_at DATETIME NOT NULL,
    finished_at DATETIME,
    PRIMARY KEY (id),
    UNIQUE (job_id, fingerprint)
);
CREATE TABLE generation_artifacts (
    id INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    job_id INTEGER,
    revision INTEGER,
    input_fingerprint VARCHAR(64),
    video_path VARCHAR(1024) NOT NULL,
    subtitle_path VARCHAR(1024),
    manifest_json JSON NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (job_id)
);
CREATE TABLE generation_jobs (
    id INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    current_stage VARCHAR(64) NOT NULL,
    status VARCHAR(16) NOT NULL,
    progress FLOAT NOT NULL,
    stage_progress FLOAT NOT NULL,
    cancel_requested BOOLEAN NOT NULL,
    kind VARCHAR(32) DEFAULT 'full' NOT NULL,
    block_index INTEGER,
    input_revision INTEGER,
    input_snapshot JSON,
    input_fingerprint VARCHAR(64),
    plan_json JSON,
    parent_job_id INTEGER,
    recovery_message TEXT,
    started_at DATETIME,
    finished_at DATETIME,
    error_message TEXT,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(project_id) REFERENCES projects (id) ON DELETE CASCADE
);
CREATE TABLE language_requests (
    request_id VARCHAR(128) NOT NULL,
    input_fingerprint VARCHAR(64) NOT NULL,
    core_request_id VARCHAR(128) NOT NULL,
    project_id INTEGER,
    base_revision INTEGER,
    status VARCHAR(32) NOT NULL,
    owner_token VARCHAR(64) NOT NULL,
    lease_until FLOAT NOT NULL,
    created_at FLOAT NOT NULL,
    request_json JSON,
    response_json JSON NOT NULL,
    PRIMARY KEY (request_id),
    UNIQUE (core_request_id)
);
CREATE TABLE language_turns (
    request_id VARCHAR(128) NOT NULL,
    parent_request_id VARCHAR(128),
    relation VARCHAR(16),
    text TEXT NOT NULL,
    successor_request_id VARCHAR(128),
    PRIMARY KEY (request_id),
    UNIQUE (parent_request_id)
);
CREATE TABLE operation_requests (
    request_id VARCHAR(128) NOT NULL,
    canonical_request TEXT NOT NULL,
    operation_id VARCHAR(128) NOT NULL,
    operation_version INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    base_revision INTEGER NOT NULL,
    result_revision INTEGER NOT NULL,
    resolved_arguments JSON NOT NULL,
    generation_requested BOOLEAN NOT NULL,
    job_id INTEGER,
    result_ref VARCHAR(256) NOT NULL,
    result_json JSON NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (request_id),
    UNIQUE (job_id)
);
CREATE TABLE project_identities (
    id INTEGER NOT NULL,
    PRIMARY KEY (id)
);
CREATE TABLE projects (
    id INTEGER NOT NULL,
    revision INTEGER DEFAULT 1 NOT NULL,
    title VARCHAR(255) NOT NULL,
    source_script TEXT NOT NULL,
    global_visual_style TEXT,
    status VARCHAR(32) NOT NULL,
    progress FLOAT NOT NULL,
    current_stage VARCHAR(64),
    llm_provider VARCHAR(32) NOT NULL,
    llm_base_url VARCHAR(512),
    llm_model VARCHAR(128),
    image_provider VARCHAR(32) NOT NULL,
    image_model VARCHAR(128),
    voicevox_url VARCHAR(512) NOT NULL,
    voicevox_speaker_id INTEGER NOT NULL,
    voicevox_speed_scale FLOAT NOT NULL,
    voicevox_pitch_scale FLOAT NOT NULL,
    voicevox_intonation_scale FLOAT NOT NULL,
    voicevox_volume_scale FLOAT NOT NULL,
    subtitle_enabled BOOLEAN NOT NULL,
    subtitle_font_size INTEGER NOT NULL,
    subtitle_position VARCHAR NOT NULL,
    subtitle_text_color VARCHAR NOT NULL,
    subtitle_outline_color VARCHAR NOT NULL,
    subtitle_background BOOLEAN NOT NULL,
    subtitle_max_chars_per_line INTEGER NOT NULL,
    visual_focus_enabled BOOLEAN DEFAULT 0 NOT NULL,
    subtitle_mode VARCHAR(16) DEFAULT 'packed' NOT NULL,
    narration_pacing_mode VARCHAR(16) DEFAULT 'fixed' NOT NULL,
    pronunciation_overrides JSON DEFAULT '[]' NOT NULL,
    narration_sentence_pause_seconds FLOAT DEFAULT '1.5' NOT NULL,
    max_slides_per_block INTEGER DEFAULT '1' NOT NULL,
    pre_margin_seconds FLOAT NOT NULL,
    post_margin_seconds FLOAT NOT NULL,
    min_display_seconds FLOAT NOT NULL,
    output_video_path VARCHAR(1024),
    current_artifact_id INTEGER,
    output_subtitle_path VARCHAR(1024),
    use_fake_providers BOOLEAN NOT NULL,
    error_message TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id)
);
INSERT INTO projects VALUES (
    101, 1, 'fixture-project', 'fixture-script', NULL, 'pending', 0.0, NULL,
    'openai_compatible', NULL, NULL, 'openai', NULL,
    'http://127.0.0.1:50021', 1, 1.0, 0.0, 1.0, 1.0,
    1, 48, 'bottom', '#FFFFFF', '#000000', 1, 36, 0, 'packed', 'fixed', '[]',
    1.5, 1, 0.15, 1.5, 2.0, NULL, NULL, NULL, 1, NULL,
    '2026-01-02 03:04:05', '2026-01-02 03:04:05'
);
CREATE TABLE settings_revisions (
    id INTEGER NOT NULL,
    project_id INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    settings_json JSON NOT NULL,
    changed_fields JSON NOT NULL,
    restored_from_revision INTEGER,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (project_id, revision)
);
CREATE INDEX ix_generation_artifacts_project_id ON generation_artifacts (project_id);
CREATE INDEX ix_external_calls_job_id ON external_calls (job_id);
CREATE INDEX ix_operation_requests_project_id ON operation_requests (project_id);
CREATE INDEX ix_settings_revisions_project_id ON settings_revisions (project_id);
CREATE INDEX ix_blocks_project_id ON blocks (project_id);
CREATE INDEX ix_generation_jobs_project_id ON generation_jobs (project_id);
COMMIT;
