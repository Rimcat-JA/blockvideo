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
} from '@/lib/types';

const API_BASE = '/api';

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
    throw new Error(`HTTP ${res.status}: ${detail}`);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export const api = {
  /** Fetch backend health and executable availability flags. */
  health: () => request<{ status: string; ffmpeg_available: boolean; ffprobe_available: boolean }>('/health'),
  /** Fetch VOICEVOX speakers from an optional engine URL. */
  speakers: (url?: string) =>
    request<SpeakersEnvelope>(`/voicevox/speakers${url ? `?url=${encodeURIComponent(url)}` : ''}`),
  /** List persisted projects. */
  listProjects: () => request<ProjectSummary[]>('/projects'),
  /** Fetch one project and its configuration fields. */
  getProject: (id: number) => request<ProjectDetail>(`/projects/${id}`),
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
