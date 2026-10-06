import { useEffect, useState } from 'react';
import { api } from '@/api/client';
import type { StartupState } from '@/lib/types';

const POLL_INTERVAL_MS = 2000;
const NETWORK_FAILURE_MESSAGE = '起動状態を確認できません。通信を確認してから再度確認してください。';
const BACKUP_MIGRATION_GUIDANCE = 'アプリを停止したまま、文書化されたバックアップ復元手順を確認してください。復元後にアプリを再起動してください。';
const NO_BACKUP_MIGRATION_GUIDANCE = 'アプリを停止して再起動してください。解決しない場合はサポートに連絡してください。';

export function StartupStatus() {
  const [startup, setStartup] = useState<StartupState | null>(null);
  const [failed, setFailed] = useState(false);
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    const fetchStatus = async () => {
      try {
        const state = await api.startup(controller.signal);
        if (controller.signal.aborted) return;
        setStartup(state);
        setFailed(false);
        if (state.status === 'starting') {
          timer = setTimeout(() => { void fetchStatus(); }, POLL_INTERVAL_MS);
        }
      } catch {
        if (!controller.signal.aborted) setFailed(true);
      }
    };

    void fetchStatus();
    return () => {
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [requestVersion]);

  if (failed) {
    return (
      <div role="alert" className="border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
        <p>{NETWORK_FAILURE_MESSAGE}</p>
        <button type="button" className="btn-secondary mt-2" onClick={() => {
          setStartup(null);
          setFailed(false);
          setRequestVersion((value) => value + 1);
        }}>起動状態を再確認</button>
      </div>
    );
  }
  if (!startup) {
    return <div role="status" className="text-sm text-slate-600">起動状態を確認しています。</div>;
  }
  if (startup.status === 'migration_failed') {
    return (
      <div role="alert" className="border border-red-300 bg-red-50 p-3 text-sm text-red-900">
        <p>{startup.message}</p>
        <p className="mt-1">
          {startup.backup_available ? BACKUP_MIGRATION_GUIDANCE : NO_BACKUP_MIGRATION_GUIDANCE}
        </p>
      </div>
    );
  }
  if (startup.status === 'starting') {
    return <div role="status" className="text-sm text-slate-600">{startup.message}起動が完了するまでお待ちください。</div>;
  }
  return <div role="status" className="text-sm text-slate-600">{startup.message}</div>;
}
