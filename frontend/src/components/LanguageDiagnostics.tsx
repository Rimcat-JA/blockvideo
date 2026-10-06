/** Local-only request inspector: raw text is never part of a logging/export API. */
import type { LanguageResponse } from '@/lib/language-types';

export function LanguageDiagnostics({ response }: { response: LanguageResponse }) {
  const timing = response.diagnostics;
  const retrieval = timing?.retrieval;
  const proposed = response.interpretation?.proposal;
  const result = response.generation_result ?? response.result;
  const failure = response.failure ?? response.interpretation?.failure;
  const duration = (value: number | null | undefined) => value == null ? '未記録' : `${(value / 1000).toFixed(2)}秒`;
  return <details className="mt-4 text-xs text-slate-500">
    <summary className="cursor-pointer">要求の記録</summary>
    <p className="mt-2 break-all">要求ID: {response.request_id}</p>
    <p className="break-all">実行要求ID: {response.core_request_id}</p>
    {response.parent_request_id && <p className="break-all">関連する要求ID: {response.parent_request_id}（{response.relation}）</p>}
    <p>設定の版: {response.base_revision ?? '不明'} → {result?.revision ?? '確定した変更なし'}</p>
    <p>依頼の判断・検証: {duration(timing?.interpretation_ms)} / 操作の受付: {duration(timing?.execution_ms)}</p>
    {timing?.generation_execution_ms != null && <p>後続の生成受付: {duration(timing.generation_execution_ms)}</p>}
    <p>上記には、確認ボタンを押すまでの時間と動画生成の時間を含みません。</p>
    <p>モデル呼出し: {retrieval?.chat_calls ?? response.interpretation?.attempts ?? '未記録'}回</p>
    {retrieval && <div className="mt-2 space-y-1" aria-label="候補検索の記録">
      <p>検索の数値はコサイン類似度です。正解の確率ではなく、操作を実行する許可でもありません。</p>
      <p>候補の拡張: {retrieval.expansion_count}回 / 全操作の確認: {retrieval.all_tools_count}回</p>
      <p>文章のベクトル化: {retrieval.embedding_calls}回・{duration(retrieval.embedding_ms)}</p>
      <p>検索・判断の合計: {duration(retrieval.elapsed_ms)} / 検索の状態: {retrieval.reason}</p>
      <ol className="list-inside list-decimal break-all">{retrieval.ranking.map(c => <li key={`${c.operation_id}@${c.operation_version}`}>
        {c.operation_id} v{c.operation_version}: {c.score.toFixed(4)}
      </li>)}</ol>
      {retrieval.stages.map(s => <div key={s.name} className="break-all"><p>{({ initial: '最初の候補', expanded: '候補を拡張', wide: '上位32件まで拡張', all_tools: '全操作' })[s.name]}:
        {s.candidates.length}件・呼出し{s.chat_calls}回・{duration(s.elapsed_ms)}・結果 {s.result}・入力{s.request_bytes} / 出力{s.response_bytes}バイト</p>
        {s.candidate_state && <p>この段階で残した実行不可の候補: {s.candidate_state.candidates.filter(c => c.readiness === 'blocked').length}件（判断時点の記録）</p>}
      </div>)}
      <p>バイト数は送受信した内容の大きさです。トークン数・料金ではありません。</p>
    </div>}
    {(response.failure || response.interpretation?.failure) && <p className="break-all">失敗理由: {(response.failure ?? response.interpretation?.failure)?.reason_code}</p>}
    {failure?.http_status != null && <p>モデル接続のHTTP状態: {failure.http_status}</p>}
    {timing?.guard_code && <p>保存前の確認理由: {timing.guard_code}</p>}
    {result?.job_id != null && <p>生成ジョブ: {result.job_id}</p>}
    {timing?.candidates && <p className="mt-2 break-all">提示した候補: {timing.candidates.map(c => `${c.operation_id} v${c.operation_version}`).join(' / ') || '未記録'}</p>}
    {proposed?.kind === 'operation' && <><p className="mt-2 break-all">抽出した操作: {proposed.operation_id} v{proposed.operation_version}</p>
      <pre className="mt-1 whitespace-pre-wrap break-all">{JSON.stringify(proposed.arguments, null, 2)}</pre></>}
    {result && <><p className="mt-2">確定した値（実行記録）</p><pre className="mt-1 whitespace-pre-wrap break-all">{JSON.stringify(result.resolved_arguments, null, 2)}</pre></>}
    <p className="mt-2">この詳細には設定内容が含まれます。アプリの操作ログには依頼文や読み方の本文を出力しません。</p>
  </details>;
}
