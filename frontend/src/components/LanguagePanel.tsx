/** Project-scoped linked clarification/correction; generation confirmation stays explicit. */
import { useEffect, useRef, useState } from 'react';
import type { LanguageController } from '@/api/useLanguageRequest';
import type { ProjectDetail, ProjectHistory } from '@/lib/types';
import { LanguageResultCard } from '@/components/LanguageResultCard';
import { LanguageWaiting } from '@/components/LanguageWaiting';
import { LanguageYoloReport } from '@/components/LanguageYoloReport';
import { LanguagePlanCard } from '@/components/LanguagePlanCard';

const YOLO_KEY = 'blockvideo.language.yolo';

function readYolo(): boolean {
  try { return localStorage.getItem(YOLO_KEY) === 'on'; } catch { return false; }
}

function saveYolo(value: boolean): void {
  try { localStorage.setItem(YOLO_KEY, value ? 'on' : 'off'); } catch { /* per-viewer convenience only */ }
}

export function LanguagePanel({ project, history, controller, disabled, unavailable, running, yoloAvailable = false }: {
  project: ProjectDetail; history?: ProjectHistory; controller: LanguageController;
  disabled: boolean; unavailable: boolean; running: boolean;
  /** The server accepts unattended (YOLO) requests; the toggle itself defaults off. */
  yoloAvailable?: boolean;
}) {
  const [text, setText] = useState('');
  const [mode, setMode] = useState<'new' | 'answer' | 'correction'>('new');
  const [yoloChosen, setYoloChosen] = useState(readYolo);
  const yolo = yoloAvailable && yoloChosen && mode === 'new';
  const input = useRef<HTMLTextAreaElement>(null);
  const { session, response, busy, issue, uncertain, locked } = controller;
  const eligible = !!response?.dialogue_available && !response.superseded_by;
  const canAnswer = eligible && response?.status === 'needs_input';
  const canCorrect = eligible && ['needs_input', 'ready', 'completed'].includes(response?.status ?? '');
  const canDismiss = eligible && ['needs_input', 'ready'].includes(response?.status ?? '');
  useEffect(() => {
    setMode(response?.dialogue_available && !response.superseded_by && response.status === 'needs_input' ? 'answer'
      : response?.dialogue_available && !response.superseded_by && response.status === 'ready' ? 'correction' : 'new');
  }, [response?.request_id, response?.status, response?.dialogue_available, response?.superseded_by]);
  const edit = () => { setMode(canCorrect ? 'correction' : 'new'); setText(session?.request.text ?? ''); input.current?.focus(); };
  const send = () => {
    if (disabled || unavailable) return;
    const sent = mode === 'new' ? controller.submit(text, project.revision, yolo) : controller.continueRequest(text, project.revision, mode);
    if (sent) setText('');
  };
  return <section className="mt-6 rounded-xl border border-indigo-200 bg-indigo-50/40 p-4 sm:p-5" aria-label="自然言語で操作">
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div><h2 className="text-base font-semibold text-slate-900">言葉で操作する</h2>
        <p className="mt-1 text-sm text-slate-600">対象: <strong>{project.title}</strong> · 設定の版 {project.revision}</p></div>
      <span className={yolo ? 'rounded-full bg-amber-100 px-3 py-1 text-xs text-amber-900' : 'rounded-full bg-white px-3 py-1 text-xs text-indigo-700'}>
        {yolo ? 'YOLO：確認なしで最後まで実行' : '設定の保存と動画生成を分けて確認'}</span>
    </div>
    {eligible && !locked && <div className="mt-3 flex flex-wrap gap-2" aria-label="依頼の続け方">
      <button type="button" className="btn-secondary" aria-pressed={mode === 'new'} onClick={() => { setMode('new'); setText(''); }}>新しい依頼</button>
      {canAnswer && <button type="button" className="btn-secondary" aria-pressed={mode === 'answer'} onClick={() => { setMode('answer'); setText(''); input.current?.focus(); }}>質問に回答</button>}
      {canCorrect && <button type="button" className="btn-secondary" aria-pressed={mode === 'correction'} onClick={() => { setMode('correction'); setText(''); input.current?.focus(); }}>この依頼を訂正</button>}
      {canDismiss && <button type="button" className="btn-secondary" disabled={disabled || unavailable} onClick={() => {
        if (controller.continueRequest('この依頼を取り下げる', project.revision, 'dismiss')) setText('');
      }}>この依頼を取り下げる</button>}
    </div>}
    {mode !== 'new' && response && <div className="mt-3 rounded border border-indigo-100 bg-white p-3 text-sm text-slate-700">
      <p className="whitespace-pre-wrap break-words">{mode === 'answer' ? '回答先の依頼' : '訂正する依頼'}: {session?.request.text}</p>
      {response.clarification && <p className="mt-1">質問: {response.clarification.question}</p>}
      <p className="mt-1 text-xs text-slate-500">{mode === 'answer' ? '質問に沿って回答してください。依頼全体の確認を求められた場合は、希望をまとめて入力してください。' : '設定の訂正は、現在保存済みの値を基準にします。動画生成は確認後に開始します。'}</p>
    </div>}
    <form className="mt-4" onSubmit={(event) => { event.preventDefault(); send(); }}>
      <label className="label" htmlFor={`language-input-${project.id}`}>{mode === 'new' ? 'この動画への依頼' : mode === 'answer' ? '確認への回答' : '訂正内容'}</label>
      <textarea ref={input} id={`language-input-${project.id}`} className="input mt-1 min-h-24 resize-y" rows={3}
        placeholder={mode === 'answer' ? '例: 56px / エーピーアイ' : mode === 'correction' ? '例: 違う、少し小さく / 60pxにして' : '例: 字幕を56pxにして / 動画の状態を教えて / 今の設定で作り直して'}
        maxLength={2000} value={text} onChange={(event) => setText(event.target.value)}
        disabled={busy || uncertain} aria-describedby={`language-help-${project.id}`} />
      <div className="mt-2 flex flex-wrap items-center justify-between gap-3">
        <p id={`language-help-${project.id}`} className="text-xs text-slate-500">{yolo
          ? '確認や質問をせず、書かれていない値は推測して、設定の保存から動画の生成まで続けて実行します。'
          : '設定は送信後に保存します。動画生成は確認後に開始します。'}</p>
        <button type="submit" className="btn-primary" disabled={disabled || unavailable || locked || !text.trim()}>
          {busy ? session?.action === 'confirm' ? '開始要求を確認中…' : yolo ? '自動実行中…' : '依頼を確認中…'
            : mode === 'answer' ? '回答を送信' : mode === 'correction' ? '訂正を送信' : yolo ? '確認なしで実行' : '依頼を送信'}
        </button>
      </div>
      {yoloAvailable && mode === 'new' && <label className="mt-2 flex items-center gap-2 text-sm text-slate-700">
        <input type="checkbox" checked={yoloChosen} disabled={busy}
          onChange={(event) => { setYoloChosen(event.target.checked); saveYolo(event.target.checked); }} />
        確認なしで最後まで実行（YOLO）
      </label>}
    </form>
    {running && <p className="mt-3 text-sm text-amber-800">生成中です。状態の確認はできますが、設定変更は完了または停止後に行ってください。</p>}
    {session && <p className="mt-3 whitespace-pre-wrap break-words text-sm text-slate-600">今回の依頼: {session.request.text}</p>}
    {session?.parentPreview && <details className="mt-2 text-sm text-slate-500"><summary>関連する前の依頼</summary>
      <p className="whitespace-pre-wrap break-words">{session.parentPreview.text}</p>{session.parentPreview.question && <p>{session.parentPreview.question}</p>}
    </details>}
    <LanguageWaiting controller={controller} />
    {issue && <p role="alert" className="mt-3 break-words text-sm text-red-700">{issue}</p>}
    {!busy && uncertain && <div className="mt-3 rounded border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
      <p>結果がまだ確定していません。新しい依頼を送る前に、この要求の結果を確認してください。</p>
      <div className="mt-2 flex flex-wrap gap-2">
        <button type="button" className="btn-secondary" onClick={controller.lookup}>結果を照会する</button>
        <button type="button" className="btn-secondary" onClick={controller.resend}>同じ要求を再送する</button>
      </div>
    </div>}
    {response && <LanguageYoloReport response={response} />}
    {response?.plan && <LanguagePlanCard response={response} project={project} busy={busy}
      disabled={disabled || running} unavailable={unavailable}
      onConfirm={() => controller.confirm(project.revision)} onEdit={edit} />}
    {response && !response.plan && <LanguageResultCard response={response} project={project} history={history}
      busy={busy} disabled={disabled || running} unavailable={unavailable}
      onConfirm={() => controller.confirm(project.revision)} onEdit={edit} />}
  </section>;
}
