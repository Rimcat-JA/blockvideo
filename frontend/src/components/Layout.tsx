/** Shared page shell with navigation and the centered content column. */
import type { ReactNode } from 'react';

/** Content supplied to the application shell. */
interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  /** Render navigation around one route's page content. */
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <a href="/" className="text-lg font-bold text-accent-700">
            BlockVideo
          </a>
          <nav className="space-x-4 text-sm">
            <a href="/" className="text-slate-600 hover:text-slate-900">
              プロジェクト一覧
            </a>
            <a href="/projects/new" className="text-slate-600 hover:text-slate-900">
              新規作成
            </a>
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  );
}
