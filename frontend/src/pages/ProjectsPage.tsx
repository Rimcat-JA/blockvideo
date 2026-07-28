import { Link } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { useProjects, useDeleteProject } from '@/api/hooks';
import { StatusBadge } from '@/components/StatusBadge';

export function ProjectsPage() {
  const { data, isLoading, error } = useProjects();
  const del = useDeleteProject();

  return (
    <Layout>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-800">プロジェクト一覧</h1>
        <div className="flex items-center gap-3">
          <Link to="/" className="btn-primary">
            ⚡ クイック生成
          </Link>
          <Link to="/projects/new" className="btn-secondary">
            詳細設定で作成
          </Link>
        </div>
      </div>

      <div className="mt-6 space-y-3">
        {isLoading && <p className="text-sm text-slate-500">読み込み中...</p>}
        {error && (
          <p className="text-sm text-red-600">読み込みに失敗しました: {(error as Error).message}</p>
        )}
        {data && data.length === 0 && (
          <div className="rounded-md border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500">
            まだプロジェクトがありません。「新規作成」から始めてください。
          </div>
        )}
        {data?.map((p) => (
          <article
            key={p.id}
            className="flex items-center justify-between rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm"
          >
            <div>
              <Link to={`/projects/${p.id}`} className="text-base font-semibold text-accent-700 hover:underline">
                {p.title}
              </Link>
              <p className="text-xs text-slate-500">
                {p.block_count} blocks · 進捗 {Math.round((p.progress || 0) * 100)}%
              </p>
            </div>
            <div className="flex items-center gap-3">
              <StatusBadge status={p.status} />
              <button
                type="button"
                className="btn-danger"
                onClick={() => {
                  if (window.confirm(`「${p.title}」を削除しますか？`)) {
                    del.mutate(p.id);
                  }
                }}
              >
                削除
              </button>
            </div>
          </article>
        ))}
      </div>
    </Layout>
  );
}