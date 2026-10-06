import { useState } from 'react';
import type { SettingsVersion } from '@/lib/types';
import { settingNames as names, displaySetting as display } from '@/lib/settings-display';

export function SettingsHistory({ versions, revision, disabled, onRestore }: {
  versions: SettingsVersion[]; revision: number; disabled: boolean; onRestore: (revision: number) => void;
}) {
  const [selected, setSelected] = useState('');
  const version = versions.find((item) => String(item.revision) === selected);
  return (
    <section id="settings-history" className="mt-6 rounded-lg border border-slate-200 bg-white p-4">
      <h2 className="text-base font-semibold text-slate-800">設定の履歴</h2>
      <p className="mt-1 text-sm text-slate-500">保存した任意の版を選んで、現在の設定として保存できます。生成は自動で開始しません。台本やブロック編集はこの操作では戻しません。</p>
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <label className="label grow">戻す設定の版
          <select className="input mt-1" value={selected} disabled={disabled} onChange={(event) => setSelected(event.target.value)}>
            <option value="">版を選択してください</option>
            {versions.map((item) => <option key={item.revision} value={item.revision}>
              版 {item.revision} · {new Date(item.created_at).toLocaleString('ja-JP')}{item.revision === revision ? '（現在）' : ''}
            </option>)}
          </select>
        </label>
        <button className="btn-secondary" type="button" disabled={disabled || !version || version.revision === revision}
          onClick={() => version && onRestore(version.revision)}>選んだ設定に戻す</button>
      </div>
      {version && <details className="mt-3 text-sm text-slate-600">
        <summary className="cursor-pointer">選んだ設定の内容（版 {version.revision}）</summary>
        {version.restored_from_revision != null && <p className="mt-2">版 {version.restored_from_revision} から戻した設定です。</p>}
        <dl className="mt-2 grid gap-x-4 gap-y-1 sm:grid-cols-[12rem_1fr]">
          {Object.entries(version.settings).filter(([key]) => key in names).map(([key, value]) => (
            <div key={key} className="contents"><dt>{names[key]}</dt><dd className="break-words">{display(value)}</dd></div>
          ))}
        </dl>
      </details>}
    </section>
  );
}
