import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import { connectionMessages, operationModeLabels } from '@/lib/language-connection';

const queryKey = ['language-connection'];

export function LanguageConnection() {
  const client = useQueryClient();
  const configuration = useQuery({ queryKey, queryFn: () => api.languageConnection(),
    retry: false, staleTime: Infinity, refetchOnWindowFocus: false });
  const check = useMutation({ mutationFn: () => api.languageConnection(true), retry: false,
    onSuccess: (value) => client.setQueryData(queryKey, value) });
  const value = configuration.data;
  return <div>
    <p className="mt-3 text-sm text-slate-600">操作の判断方式：{value?.operation_mode
      ? operationModeLabels[value.operation_mode] : '未確認'}（接続先の設定）</p>
    <details className="mt-2 rounded-lg border border-slate-200 bg-white p-3 text-sm text-slate-600">
    <summary className="cursor-pointer break-words font-medium text-slate-800">言葉で操作するAI：{value?.model ?? '接続設定'}</summary>
    <div className="mt-3 space-y-2 break-words">
      {value?.base_url && <p>このPCの接続先：{value.base_url}</p>}
      <p aria-live="polite">{configuration.error || check.error ? '接続情報を取得できません。もう一度確認してください。'
        : check.isPending ? '接続を確認中…' : value ? connectionMessages[value.status] : '設定を読み込み中…'}</p>
      <p>接続確認では依頼文を送りません。接続できない場合に、別のAIへ自動で切り替えることはありません。</p>
      <button type="button" className="btn-secondary" disabled={configuration.isPending || check.isPending}
        onClick={() => check.mutate()}>接続を確認</button>
    </div>
  </details></div>;
}
