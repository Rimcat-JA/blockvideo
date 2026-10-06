/** Browser reload identity; no cached response is trusted as proof of success. */
import type { LanguageConfirmation, LanguageRequest } from '@/lib/language-types';

export interface LanguageSession {
  request: LanguageRequest;
  epoch: string;
  action: 'submit' | 'confirm' | 'lookup';
  confirmation?: LanguageConfirmation;
  parentPreview?: { text: string; question?: string };
  startedAt?: number;
}

const key = (projectId: number) => `blockvideo-language-${projectId}`;

export function readLanguageSession(projectId: number): LanguageSession | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key(projectId)) ?? 'null') as LanguageSession | null;
    if (!value || value.request?.target?.selected_project_id !== projectId
      || typeof value.request.request_id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value.request.request_id)
      || typeof value.request.text !== 'string' || !value.request.text.trim() || value.request.text.length > 2000
      || !Number.isInteger(value.request.base_revision) || value.request.base_revision < 1
      || typeof value.epoch !== 'string' || !['submit', 'confirm', 'lookup'].includes(value.action)) return null;
    if (value.action === 'confirm' && (!value.confirmation
      || !/^[0-9a-f]{64}$/.test(value.confirmation.confirmation_token)
      || typeof value.confirmation.confirm_generation !== 'boolean')) return null;
    if (value.request.mode !== undefined && !['normal', 'yolo'].includes(value.request.mode)) return null;
    const parent = value.request.continuation;
    if (parent && (typeof parent.parent_request_id !== 'string'
      || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(parent.parent_request_id)
      || !['answer', 'correction', 'dismiss'].includes(parent.relation))) return null;
    if (value.parentPreview && (typeof value.parentPreview.text !== 'string' || value.parentPreview.text.length > 2000
      || (value.parentPreview.question != null && (typeof value.parentPreview.question !== 'string' || value.parentPreview.question.length > 240)))) return null;
    return { ...value, startedAt: typeof value.startedAt === 'number' && Number.isFinite(value.startedAt)
      && value.startedAt > 0 && value.startedAt <= Date.now() ? value.startedAt : undefined };
  } catch { return null; }
}

export function saveLanguageSession(projectId: number, session: LanguageSession, expectedEpoch?: string): boolean {
  if (expectedEpoch && readLanguageSession(projectId)?.epoch !== expectedEpoch) return false;
  sessionStorage.setItem(key(projectId), JSON.stringify(session));
  return true;
}
