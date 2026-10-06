/** Typed HTTP boundary for the FastAPI `/api` routes. */
import type {
  BlockSummary,
  CreateProjectInput,
  JobSummary,
  ProjectDetail,
  ProjectSummary,
  SpeakersEnvelope,
  QuickCreateInput,
  QuickCreateResponse,
  OperationRequest,
  OperationResult,
  ProjectHistory,
  StartupState,
} from '@/lib/types';

const API_BASE = '/api';
import type { CandidateReadinessSnapshot, LanguageConfirmation, LanguageRequest, LanguageResponse } from '@/lib/language-types';
import type { LanguageConnection } from '@/lib/language-connection';

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
  signal?: AbortSignal,
): Promise<T> {
  /** Execute a JSON request and turn non-2xx responses into Error objects. */
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
    signal,
  });
  if (!res.ok) {
    const detail = await res.text();
    let message = detail;
    try {
      const parsed = JSON.parse(detail).detail;
      if (typeof parsed === 'string') message = parsed;
      else if (typeof parsed?.message === 'string') message = parsed.message;
    } catch { /* Non-JSON errors retain the server message. */ }
    throw new ApiError(res.status, `HTTP ${res.status}: ${message}`);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export const api = {
  startup: (signal?: AbortSignal) => request<StartupState>('/startup', undefined, signal),
  languageConnection: (check = false) =>
    request<LanguageConnection>(`/language/connection${check ? '?check=true' : ''}`),
  submitLanguage: (input: LanguageRequest, signal?: AbortSignal) =>
    request<LanguageResponse>('/language/requests', { method: 'POST', body: JSON.stringify(input) }, signal),
  getLanguageRequest: (requestId: string, signal?: AbortSignal) =>
    request<LanguageResponse>(`/language/requests/${encodeURIComponent(requestId)}`, undefined, signal),
  getCandidateReadiness: (requestId: string, signal?: AbortSignal) =>
    request<CandidateReadinessSnapshot>(`/language/requests/${encodeURIComponent(requestId)}/candidate-readiness`, undefined, signal),
  confirmLanguage: (requestId: string, input: LanguageConfirmation, signal?: AbortSignal) =>
    request<LanguageResponse>(`/language/requests/${encodeURIComponent(requestId)}/execute`,
      { method: 'POST', body: JSON.stringify(input) }, signal),
  /** Fetch backend health and executable availability flags. */
  health: () => request<{ status: string; ffmpeg_available: boolean; ffprobe_available: boolean }>('/health'),
  /** Fetch VOICEVOX speakers from an optional engine URL. */
  speakers: (url?: string) =>
    request<SpeakersEnvelope>(`/voicevox/speakers${url ? `?url=${encodeURIComponent(url)}` : ''}`),
  /** List persisted projects. */
  listProjects: () => request<ProjectSummary[]>('/projects'),
  /** Fetch one project and its configuration fields. */
  getProject: (id: number) => request<ProjectDetail>(`/projects/${id}`),
  getProjectHistory: (id: number) => request<ProjectHistory>(`/projects/${id}/history`),
  executeOperation: (input: OperationRequest) =>
    request<OperationResult>('/operations/execute', { method: 'POST', body: JSON.stringify(input) }),
  /** Create a project without starting generation. */
  createProject: (input: CreateProjectInput) =>
    request<ProjectDetail>('/projects', { method: 'POST', body: JSON.stringify(input) }),
  /** Create and immediately queue a project from pasted script text. */
  quickCreate: (input: QuickCreateInput) =>
    request<QuickCreateResponse>('/projects/quick', {
      method: 'POST',
      body: JSON.stringify(input),
    }),
  /** Delete a project and its generated artifacts. */
  deleteProject: (id: number) =>
    request<void>(`/projects/${id}`, { method: 'DELETE' }),
  /** List the generated blocks belonging to a project. */
  listBlocks: (projectId: number) => request<BlockSummary[]>(`/projects/${projectId}/blocks`),
  /** Queue a full split-to-MP4 generation run. */
  generateAll: (projectId: number) =>
    request<{ job: JobSummary; message: string }>(`/projects/${projectId}/generate-all`, {
      method: 'POST',
    }),
  /** Request cancellation of active project jobs. */
  cancelProject: (projectId: number) =>
    request<{ cancelled: number }>(`/projects/${projectId}/cancel`, { method: 'POST' }),
  /** Queue a render-only rebuild using existing block assets. */
  rerender: (projectId: number) =>
    request<{ job: JobSummary; message: string }>(`/projects/${projectId}/rerender`, {
      method: 'POST',
    }),
  /** Queue visual regeneration for one block. */
  regenerateBlockVisual: (blockId: number) =>
    request<{ job: JobSummary; message: string }>(`/blocks/${blockId}/regenerate-visual`, {
      method: 'POST',
    }),
  /** Queue audio regeneration for one block. */
  regenerateBlockAudio: (blockId: number) =>
    request<{ job: JobSummary; message: string }>(`/blocks/${blockId}/regenerate-audio`, {
      method: 'POST',
    }),
  /** Queue a project-level rerender from a block action. */
  rerenderBlock: (blockId: number) =>
    request<{ job: JobSummary; message: string }>(`/blocks/${blockId}/rerender`, {
      method: 'POST',
    }),
  artifactUrl(projectId: number, kind: 'image' | 'audio' | 'video', blockIndex: number): string {
    /** Build an API URL for a block artifact without fetching it. */
    return `${API_BASE}/projects/${projectId}/artifacts/${kind}/${blockIndex}`;
  },
  downloadUrl(projectId: number): string {
    /** Build the API URL for the final project MP4. */
    return `${API_BASE}/projects/${projectId}/download`;
  },
};
