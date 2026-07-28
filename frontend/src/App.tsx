/** Route table for the BlockVideo single-page application. */
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ProjectsPage } from '@/pages/ProjectsPage';
import { NewProjectPage } from '@/pages/NewProjectPage';
import { ProjectDetailPage } from '@/pages/ProjectDetailPage';
import { QuickGeneratePage } from '@/pages/QuickGeneratePage';

export default function App() {
  /** Render the route selected by the browser URL. */
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<QuickGeneratePage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/new" element={<NewProjectPage />} />
        <Route path="/projects/:id" element={<ProjectDetailPage />} />
      </Routes>
    </BrowserRouter>
  );
}
