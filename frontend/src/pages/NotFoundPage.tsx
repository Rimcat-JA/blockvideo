/** Fallback for unknown paths and malformed project ids. */
import { Link } from 'react-router-dom';
import { Layout } from '@/components/Layout';

export function NotFoundPage() {
  /** Keep the header and a way back instead of rendering a blank page. */
  return (
    <Layout>
      <h1 className="text-2xl font-bold text-slate-800">ページが見つかりません</h1>
      <p className="mt-2 text-sm text-slate-600">URL が正しいか確認してください。</p>
      <Link to="/projects" className="mt-4 inline-block text-sm text-accent-700 hover:underline">
        プロジェクト一覧へ戻る
      </Link>
    </Layout>
  );
}
