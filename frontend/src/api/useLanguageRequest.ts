/** One frozen intent, explicit confirmation, GET-only recovery, fenced responses. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { api, ApiError } from '@/api/client';
import { readLanguageSession, saveLanguageSession, type LanguageSession } from '@/api/language-storage';
import type { LanguageResponse } from '@/lib/language-types';

export function useLanguageRequest(projectId: number) {
  const qc = useQueryClient();
  const [session, setSession] = useState(() => readLanguageSession(projectId));
  const [response, setResponse] = useState<LanguageResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [issue, setIssue] = useState<string | null>(null);
  const [rejected, setRejected] = useState(false);
  const [activity, setActivity] = useState<'submit' | 'confirm' | 'lookup' | null>(null);
  const [activityStartedAt, setActivityStartedAt] = useState<number | null>(null);
  const current = useRef(session);
  const flight = useRef(false);
  const mounted = useRef(true);

  const refresh = useCallback(() => Promise.allSettled([
    qc.invalidateQueries({ queryKey: ['project', projectId] }),
    qc.invalidateQueries({ queryKey: ['history', projectId] }),
    qc.invalidateQueries({ queryKey: ['blocks', projectId] }),
  ]), [projectId, qc]);

  const perform = useCallback(async (entry: LanguageSession, lookup: boolean) => {
    if (flight.current) return;
    flight.current = true;
    setBusy(true);
    setActivity(lookup ? 'lookup' : entry.action);
    setActivityStartedAt(Date.now());
    setIssue(null);
    setRejected(false);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), lookup ? 15000 : 180000);
    const stillCurrent = () => mounted.current && current.current?.epoch === entry.epoch
      && readLanguageSession(projectId)?.epoch === entry.epoch;
    try {
      const result = lookup || entry.action === 'lookup'
        ? await api.getLanguageRequest(entry.request.request_id, controller.signal)
        : entry.action === 'confirm'
          ? await api.confirmLanguage(entry.request.request_id, entry.confirmation!, controller.signal)
          : await api.submitLanguage(entry.request, controller.signal);
      if (result.request_id !== entry.request.request_id || result.project_id !== projectId
        || (result.result && result.result.project_id !== projectId)
        || (result.prepared_request && result.prepared_request.target.project_id !== projectId)
        || (result.generation_request && result.generation_request.target.project_id !== projectId)
        || (result.generation_result && result.generation_result.project_id !== projectId)) {
        throw new Error('対象または要求の照合に失敗しました。結果をもう一度確認してください。');
      }
      if (!stillCurrent()) return;
      setResponse(result);
      // A ready non-confirming proposal may be a crash between prepare and core
      // commit. Preserve the submit body so only an explicit resend executes it.
      const settled = result.status !== 'interpreting'
        && !(result.status === 'ready' && !result.requires_confirmation);
      if (settled) {
        const next = { ...entry, action: 'lookup' as const, confirmation: undefined };
        current.current = next;
        setSession(next);
        try { saveLanguageSession(projectId, next, entry.epoch); }
        catch { setIssue('結果の控えを更新できませんでした。再読み込み時はサーバーの結果を照会します。'); }
      }
      await refresh();
    } catch (cause) {
      if (!stillCurrent()) return;
      const notFound = lookup && cause instanceof ApiError && cause.status === 404;
      const knownRejection = (notFound && entry.action === 'lookup')
        || (!lookup && cause instanceof ApiError && [400, 401, 403, 404, 409, 422].includes(cause.status));
      setRejected(knownRejection);
      setIssue(notFound
        ? entry.action === 'lookup'
          ? 'サーバーに要求の記録が見つかりません。現在の状態を確認してから、新しく依頼してください。'
          : 'サーバーに要求が見つかりません。同じ要求を再送して確認してください。'
        : (cause instanceof ApiError && cause.status >= 500) || cause instanceof TypeError
          || (cause instanceof Error && cause.name === 'AbortError')
          ? '応答を確認できませんでした。処理が完了している可能性があるため、結果の照会または同じ要求の再送で確認してください。'
        : cause instanceof Error ? cause.message : '応答を確認できませんでした。');
      // The mutation may have committed before its acknowledgement was lost.
      // Refresh observed state without claiming this request has succeeded.
      await refresh();
    } finally {
      clearTimeout(timer);
      flight.current = false;
      if (mounted.current) { setBusy(false); setActivity(null); }
    }
  }, [projectId, refresh]);

  useEffect(() => {
    mounted.current = true;
    const saved = current.current;
    if (saved) void perform(saved, true);
    return () => { mounted.current = false; };
  }, [perform]);

  useEffect(() => {
    if (response?.status !== 'interpreting' || issue || busy) return;
    const timer = setTimeout(() => { if (current.current) void perform(current.current, true); }, 2000);
    return () => clearTimeout(timer);
  }, [response, issue, busy, perform]);

  const uncertain = session != null && !rejected && (response == null || response.status === 'interpreting'
    || (response.status === 'ready' && !response.requires_confirmation)
    || (session.action === 'confirm' && !!issue));
  const locked = busy || uncertain;

  const begin = useCallback((entry: LanguageSession) => {
    entry = { ...entry, startedAt: Date.now() };
    try {
      saveLanguageSession(projectId, entry);
    } catch {
      setIssue('要求の控えを保存できません。ブラウザーの保存設定を確認してください。送信はしていません。');
      return false;
    }
    current.current = entry;
    setSession(entry);
    setResponse(null);
    setRejected(false);
    void perform(entry, false);
    return true;
  }, [perform, projectId]);

  const submit = useCallback((text: string, revision: number, yolo = false) => {
    if (flight.current || locked || !text.trim() || text.length > 2000) return false;
    return begin({ request: { request_id: crypto.randomUUID(), text: text.trim(),
      target: { selected_project_id: projectId }, base_revision: revision, ...(yolo ? { mode: 'yolo' as const } : {}) },
    epoch: crypto.randomUUID(), action: 'submit' });
  }, [begin, locked, projectId]);

  const confirm = useCallback((revision: number) => {
    if (flight.current || !current.current || response?.status !== 'ready'
      || response.superseded_by
      || !response.requires_confirmation || !response.confirmation_token
      || revision !== (response.generation_request?.base_revision ?? response.base_revision)) return;
    const generationIds = ['project.generation.start', 'project.generation.retry'];
    const generation = !!response.generation_request || generationIds.includes(response.prepared_request?.operation_id ?? '')
      || (response.plan ?? []).some((step) => generationIds.includes(step.operation_id));
    begin({ ...current.current, epoch: crypto.randomUUID(), action: 'confirm',
      confirmation: { confirmation_token: response.confirmation_token, confirm_generation: generation } });
  }, [begin, response]);

  const continueRequest = useCallback((text: string, revision: number, relation: 'answer' | 'correction' | 'dismiss') => {
    if (flight.current || locked || !response?.dialogue_available || response.superseded_by
      || !current.current || !text.trim() || text.length > 2000) return false;
    const allowed = relation === 'answer' ? ['needs_input'] : relation === 'dismiss' ? ['needs_input', 'ready'] : ['needs_input', 'ready', 'completed'];
    if (!allowed.includes(response.status)) return false;
    const expected = response.result?.revision ?? response.base_revision;
    if (relation !== 'dismiss' && expected !== revision) {
      setIssue('この質問の後に設定が変わりました。現在の設定を確認し、希望を新しく依頼してください。');
      return false;
    }
    return begin({ request: { request_id: crypto.randomUUID(), text: text.trim(),
      target: { selected_project_id: projectId }, base_revision: revision,
      continuation: { parent_request_id: response.request_id, relation } },
      parentPreview: { text: current.current.request.text, question: response.clarification?.question },
      epoch: crypto.randomUUID(), action: 'submit' });
  }, [begin, locked, projectId, response]);

  return { session, response, busy, issue, rejected, uncertain, locked, activity, activityStartedAt, submit, confirm, continueRequest,
    resend: () => { if (current.current) void perform(current.current, false); },
    lookup: () => { if (current.current) void perform(current.current, true); },
  };
}

export type LanguageController = ReturnType<typeof useLanguageRequest>;
