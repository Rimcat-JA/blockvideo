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
  health: () => request<{ status: string; ffmpeg_available: boolean; ffprobe_available: boolean }>('/health'),
  speakers: (url?: string) =>
    request<SpeakersEnvelope>(`/voicevox/speakers${url ? `?url=${encodeURIComponent(url)}` : ''}`),
  listProjects: () => request<ProjectSummary[]>('/projects'),
  getProject: (id: number) => request<ProjectDetail>(`/projects/${id}`),
  createProject: (input: CreateProjectInput) =>
    request<ProjectDetail>('/projects', { method: 'POST', body: JSON.stringify(input) }),
  quickCreate: (input: QuickCreateInput) =>
    request<QuickCreateResponse>('/projects/quick', {
      method: 'POST',
      body: JSON.stringify(input),
    }),
  deleteProject: (id: number) =>
    request<void>(`/projects/${id}`, { method: 'DELETE' }),
  listBlocks: (projectId: number) => request<BlockSummary[]>(`/projects/${projectId}/blocks`),
  generateAll: (projectId: number) =>
    request<{ job: JobSummary; message: string }>(`/projects/${projectId}/generate-all`, {
      method: 'POST',
    }),
  cancelProject: (projectId: number) =>
    request<{ cancelled: number }>(`/projects/${projectId}/cancel`, { method: 'POST' }),
  rerender: (projectId: number) =>
    request<{ job: JobSummary; message: string }>(`/projects/${projectId}/rerender`, {
      method: 'POST',
    }),
  regenerateBlockVisual: (blockId: number) =>
    request<{ job: JobSummary; message: string }>(`/blocks/${blockId}/regenerate-visual`, {
      method: 'POST',
    }),
  regenerateBlockAudio: (blockId: number) =>
    request<{ job: JobSummary; message: string }>(`/blocks/${blockId}/regenerate-audio`, {
      method: 'POST',
    }),
  rerenderBlock: (blockId: number) =>
    request<{ job: JobSummary; message: string }>(`/blocks/${blockId}/rerender`, {
      method: 'POST',
    }),
  artifactUrl(projectId: number, kind: 'image' | 'audio' | 'video', blockIndex: number): string {
    return `${API_BASE}/projects/${projectId}/artifacts/${kind}/${blockIndex}`;
  },
  downloadUrl(projectId: number): string {
    return `${API_BASE}/projects/${projectId}/download`;
  },
};