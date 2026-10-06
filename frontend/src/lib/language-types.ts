/** D17 HTTP contracts consumed by the D18 UI. */
import type { OperationRequest, OperationResult, ProjectOperationId } from '@/lib/types';

export interface LanguageRequest {
  request_id: string;
  text: string;
  target: { selected_project_id: number };
  base_revision: number;
  continuation?: { parent_request_id: string; relation: 'answer' | 'correction' | 'dismiss' };
  /** 'yolo': no confirmation or clarification; missing values are guessed and reported. */
  mode?: 'normal' | 'yolo';
}

export type YoloGuardCode = 'reference' | 'subtitle_value' | 'settings_value' | 'pending_settings' | 'reading' | 'empty_settings';

export interface YoloReport {
  guessing_allowed: boolean;
  bypassed_guards: YoloGuardCode[];
  auto_confirmed: string[];
  dropped_steps?: string[];
  unresolved: string | null;
}

export interface LanguageConfirmation {
  confirmation_token: string;
  confirm_generation: boolean;
}

export interface Clarification {
  kind: 'clarification';
  question: string;
  missing_fields: Array<'target' | 'arguments' | 'intent'>;
}

export interface CandidateReadinessSnapshot {
  observed_at: number;
  candidates: Array<{
    operation_id: string; operation_version: number;
    phase: 'candidate_preview'; arguments_checked: false;
    readiness: 'ready' | 'needs_input' | 'blocked' | 'unsupported';
    reason_code: 'target_required' | 'target_not_found' | 'project_busy' | 'arguments_unchecked' | 'operation_not_found' | null;
    missing_fields: string[]; project_id: number | null; revision: number | null;
  }>;
}

export interface LanguageResponse {
  request_id: string;
  core_request_id: string;
  mode: 'all_tools' | 'semantic';
  status: 'interpreting' | 'ready' | 'needs_input' | 'unsupported' | 'blocked' | 'error' | 'completed' | 'dismissed';
  parent_request_id?: string | null;
  relation?: 'answer' | 'correction' | 'dismiss' | null;
  superseded_by?: string | null;
  dialogue_available?: boolean;
  project_id: number | null;
  base_revision: number | null;
  interpretation: null | {
    attempts?: number;
    status: 'proposed' | 'needs_input' | 'unsupported' | 'error' | 'dismissed';
    executed: false;
    proposal: Clarification | { kind: 'unsupported' | 'no_operation'; reason: string } | {
      kind: 'operation'; operation_id: ProjectOperationId; operation_version: number; arguments: Record<string, unknown>;
    } | null;
    failure: { reason_code: string; message: string; http_status?: number | null } | null;
  };
  clarification: Clarification | null;
  prepared_request: OperationRequest | null;
  requires_confirmation: boolean;
  confirmation_token: string | null;
  result: (OperationResult & {
    project_id: number; changed: boolean; base_revision: number;
    resolved_arguments: Record<string, unknown>; generation_requested: boolean;
  }) | null;
  failure: { reason_code: string; message: string; http_status?: number | null } | null;
  executed: boolean;
  generate_after_save?: boolean;
  execution_mode?: 'normal' | 'yolo';
  yolo_report?: YoloReport | null;
  /** A multi-step plan, run in order after one confirmation (or unattended). */
  plan?: OperationRequest[] | null;
  plan_results?: Array<OperationResult & { changed?: boolean; base_revision?: number | null }>;
  generation_request?: OperationRequest | null;
  generation_result?: LanguageResponse['result'];
  diagnostics?: {
    retrieval?: {
      policy: 'semantic-3-6-all-v1' | 'semantic-5-8-all-v1' | 'all-tools-v1';
      index_sha256: string | null;
      ranking: Array<{ operation_id: string; operation_version: number; score: number; document_id: string }>;
      stages: Array<{ name: 'initial' | 'expanded' | 'wide' | 'all_tools'; candidates: Array<{ operation_id: string; operation_version: number }>;
        candidate_state?: CandidateReadinessSnapshot | null;
        result: string; chat_calls: number; elapsed_ms: number; request_bytes: number; response_bytes: number }>;
      expansion_count: number; all_tools_count: number; embedding_calls: number;
      embedding_ms: number; chat_calls: number; elapsed_ms: number; reason: string;
    } | null;
    started_at: number | null;
    candidates: Array<{ operation_id: string; operation_version: number }>;
    interpretation_ms: number | null;
    execution_ms: number | null;
    generation_execution_ms: number | null;
    guard_code: string | null;
  };
}
