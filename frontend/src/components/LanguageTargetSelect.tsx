/** Navigation changes context explicitly; pending requests stay with their project. */
import { useNavigate } from 'react-router-dom';
import { useProjects } from '@/api/hooks';

export function LanguageTargetSelect({ projectId }: { projectId: number }) {
  const projects = useProjects();
  const navigate = useNavigate();
  return <div className="mt-5 max-w-xl">
    <label className="label" htmlFor="language-project-target">対象を切り替える</label>
    <select id="language-project-target" className="input mt-1" value={projectId} disabled={!projects.data}
      onChange={(event) => navigate(`/projects/${Number(event.target.value)}`)}>
      {!projects.data?.some((project) => project.id === projectId) && <option value={projectId}>現在のプロジェクト #{projectId}</option>}
      {projects.data?.map((project) => <option key={project.id} value={project.id}>{project.title}（#{project.id}）</option>)}
    </select>
    <p className="mt-1 text-xs text-slate-500">保留中の依頼は元の対象に残ります。移動先には引き継ぎません。</p>
  </div>;
}
