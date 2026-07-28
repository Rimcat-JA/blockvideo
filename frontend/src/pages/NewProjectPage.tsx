import { Layout } from '@/components/Layout';
import { ProjectForm } from '@/components/ProjectForm';

export function NewProjectPage() {
  return (
    <Layout>
      <h1 className="text-2xl font-bold text-slate-800">新規プロジェクト</h1>
      <p className="mt-1 text-sm text-slate-600">
        完成済みの日本語台本を貼り付け、LLM/VOICEVOXの設定を入力してください。
      </p>
      <div className="mt-6">
        <ProjectForm />
      </div>
    </Layout>
  );
}