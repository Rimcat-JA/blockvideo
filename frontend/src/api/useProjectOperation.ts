/** One user intent has one durable request ID, including uncertain resends. */
import { useCallback, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '@/api/client';
import type { OperationRequest, ProjectOperationId } from '@/lib/types';

interface PendingIntent {
  label: string;
  request: OperationRequest;
}

function readPending(key: string, projectId: number): PendingIntent | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) ?? 'null') as PendingIntent | null;
    return value?.request.target.project_id === projectId && typeof value.request.request_id === 'string'
      ? value : null;
  } catch { return null; }
}

export function useProjectOperation(projectId: number) {
  const qc = useQueryClient();
  const key = `blockvideo-operation-${projectId}`;
  const [pending, setPending] = useState<PendingIntent | null>(() => readPending(key, projectId));
  const [isSending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const inFlight = useRef(false);
  const pendingRef = useRef(pending);

  const remember = useCallback((intent: PendingIntent | null, completedRequestId?: string) => {
    // An unmounted sender can finish after another mounted page has stored a new intent.
    if (intent == null && pendingRef.current?.request.request_id !== completedRequestId) return;
    pendingRef.current = intent;
    setPending(intent);
    try {
      if (intent) sessionStorage.setItem(key, JSON.stringify(intent));
      else if (readPending(key, projectId)?.request.request_id === completedRequestId) sessionStorage.removeItem(key);
    } catch { /* In-memory identity is still retained when browser storage is unavailable. */ }
  }, [key, projectId]);

  const send = useCallback(async (intent: PendingIntent) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setSending(true);
    setError(null);
    setMessage(null);
    remember(intent);
    try {
      await api.executeOperation(intent.request);
      remember(null, intent.request.request_id);
      setMessage(`${intent.label}を受け付けました。`);
    } catch (cause) {
      // A rejected 4xx has no committed effect. Network/5xx outcomes can be unknown.
      if (cause instanceof ApiError && [400, 401, 403, 404, 409, 422].includes(cause.status)) remember(null, intent.request.request_id);
      setError(cause instanceof Error ? cause.message : '応答を確認できませんでした。');
    } finally {
      await Promise.allSettled([
        qc.invalidateQueries({ queryKey: ['project', projectId] }),
        qc.invalidateQueries({ queryKey: ['history', projectId] }),
        qc.invalidateQueries({ queryKey: ['blocks', projectId] }),
      ]);
      inFlight.current = false;
      setSending(false);
    }
  }, [projectId, qc, remember]);

  const execute = useCallback((operationId: ProjectOperationId, revision: number,
    args: Record<string, unknown>, label: string) => {
    if (inFlight.current || pendingRef.current) return;
    const intent: PendingIntent = {
      label,
      request: {
        request_id: crypto.randomUUID(), operation_id: operationId, operation_version: 1,
        target: { project_id: projectId }, base_revision: revision, arguments: args,
      },
    };
    void send(intent);
  }, [projectId, send]);

  return {
    execute, isSending, error, message, pending,
    locked: isSending || pending != null,
    resend: () => { if (pendingRef.current) void send(pendingRef.current); },
  };
}
