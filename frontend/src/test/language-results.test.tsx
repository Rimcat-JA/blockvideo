import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { LanguageResultCard } from '@/components/LanguageResultCard';
import { compoundFixture, languageFixture, readyFixture } from '@/test/language-fixtures';
import { historyFixture, jobFixture, projectFixture } from '@/test/project-fixtures';
import type { LanguageResponse } from '@/lib/language-types';
import type { ProjectHistory } from '@/lib/types';

function card(response: LanguageResponse, history = historyFixture({ revision: 4, output_state: 'stale' }), revision = 4) {
  const confirm = vi.fn();
  render(<LanguageResultCard response={response} project={projectFixture({ revision, subtitle_font_size: 99 })}
    history={history} unavailable={false} busy={false} disabled={false} onConfirm={confirm} onEdit={vi.fn()} />);
  return confirm;
}

function generated(): LanguageResponse {
  const response = languageFixture();
  return { ...response, result: { ...response.result!, operation_id: 'project.generation.start',
    revision: 3, changed: false, generation_requested: true, job_id: 9, data: {} } };
}

describe('authoritative language result cards', () => {
  it('shows saved changes alongside a generation confirmation for the resulting revision', () => {
    card(compoundFixture());
    expect(screen.getByRole('status')).toHaveTextContent('設定は保存済み・動画生成は未開始');
    expect(screen.getByText('48px → 56px')).toBeInTheDocument();
    expect(screen.getByText('動画の生成 · 使用する設定の版 4')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'この内容で動画生成を開始' })).toBeEnabled();
    expect(screen.queryByText('まだ設定は変更していません。')).not.toBeInTheDocument();
  });

  it('keeps saved settings visible when the generation phase is blocked', () => {
    card({ ...compoundFixture(), status: 'blocked', requires_confirmation: false,
      failure: { reason_code: 'stale_state', message: 'stale' } }, historyFixture(), 5);
    expect(screen.getByText('48px → 56px')).toBeInTheDocument();
    expect(screen.getByText('設定の保存は完了しています。後続の動画生成は開始していません。')).toBeInTheDocument();
    expect(screen.queryByText('この依頼による設定変更・動画生成は行っていません。')).not.toBeInTheDocument();
  });

  it('uses the separate generation receipt for progress while retaining the settings diff', () => {
    const response = compoundFixture();
    card({ ...response, status: 'completed', requires_confirmation: false,
      generation_result: { ...response.result!, operation_id: 'project.generation.start', changed: false, job_id: 9 } },
    historyFixture({ revision: 4, jobs: [jobFixture({ id: 9, status: 'pending' })] }));
    expect(screen.getByRole('status')).toHaveTextContent('設定を保存し、動画生成を受け付けました');
    expect(screen.getByText('48px → 56px')).toBeInTheDocument();
    expect(screen.getByText('動画生成を受け付けました・開始待ち')).toBeInTheDocument();
    expect(screen.queryByText('この設定変更では動画生成を開始していません。')).not.toBeInTheDocument();
  });
  it('shows the saved revision diff, not the current form value or a model claim', () => {
    card(languageFixture());
    expect(screen.getByRole('status')).toHaveTextContent('設定を保存しました');
    expect(screen.getByText('48px → 56px')).toBeInTheDocument();
    expect(screen.queryByText('99px → 56px')).not.toBeInTheDocument();
    expect(screen.getByText('設定は保存済み・動画には未反映')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: '設定を戻す（履歴から選ぶ）' })).toHaveAttribute('href', '#settings-history');
  });

  it('does not claim a current video while history is older than the committed receipt', () => {
    card(languageFixture(), historyFixture({ revision: 3, output_state: 'current' }));
    expect(screen.getByText('最新の動画状態を確認中です')).toBeInTheDocument();
    expect(screen.queryByText('最新の設定が動画に反映されています')).not.toBeInTheDocument();
  });

  it('labels an unavailable before-value instead of using a possibly stale form', () => {
    card(languageFixture(), historyFixture({ revision: 4, settings_versions: [] }));
    expect(screen.getByText('変更前は未取得 → 56px')).toBeInTheDocument();
  });

  it('shows a newer revision separately from an earlier successful result', () => {
    card(languageFixture(), historyFixture({ revision: 5 }), 5);
    expect(screen.getByText('この結果の後に設定が変更されています。現在は版 5 です。')).toBeInTheDocument();
    expect(screen.getByText('48px → 56px')).toBeInTheDocument();
  });

  it('renders changed fields of an update from the committed core values', () => {
    const response = languageFixture();
    response.result = { ...response.result!, operation_id: 'project.settings.update',
      data: { settings: { subtitle_font_size: 64, voicevox_speed_scale: 1.2, title: '変更しない' }, changed_fields: ['subtitle_font_size', 'voicevox_speed_scale'] } };
    card(response);
    expect(screen.getByText('48px → 64px')).toBeInTheDocument();
    expect(screen.getByText('変更前は未取得 → 1.2')).toBeInTheDocument();
    expect(screen.queryByText('変更しない')).not.toBeInTheDocument();
  });

  it('requires an explicit click to confirm a generation proposal', () => {
    const confirm = card(readyFixture(), historyFixture(), 3);
    expect(screen.getByText('まだ動画生成は開始していません。対象と内容を確認してください。')).toBeInTheDocument();
    expect(confirm).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'この内容で動画生成を開始' }));
    expect(confirm).toHaveBeenCalledTimes(1);
  });

  it('disables confirmation when settings changed after interpretation', () => {
    const confirm = card(readyFixture());
    const button = screen.getByRole('button', { name: 'この内容で動画生成を開始' });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(confirm).not.toHaveBeenCalled();
  });

  it('labels a review-all status confirmation as an operation, not a settings save', () => {
    const response = readyFixture();
    response.prepared_request = { ...response.prepared_request!, operation_id: 'project.status.get', arguments: {} };
    card(response, historyFixture(), 3);
    expect(screen.getByRole('button', { name: 'この内容で操作を実行' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'この内容で設定を保存' })).not.toBeInTheDocument();
    expect(screen.getByText('まだこの操作は実行していません。')).toBeInTheDocument();
  });

  it('shows a queued job as pending, never as a finished video', () => {
    card(generated(), historyFixture({ jobs: [jobFixture({ id: 9, status: 'pending', progress: 0 })] }), 3);
    expect(screen.getByText('動画生成を受け付けました・開始待ち')).toBeInTheDocument();
    expect(screen.queryByText('この要求の動画が完成しました')).not.toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: '動画生成の進捗' })).toHaveAttribute('value', '0');
  });

  it('requires an available artifact from this job before reporting video completion', () => {
    card(generated(), historyFixture({ jobs: [jobFixture({ id: 9, status: 'completed', progress: 1 })] }), 3);
    expect(screen.getByText('生成処理は完了しました。動画ファイルを確認中です')).toBeInTheDocument();
    expect(screen.queryByText('この要求の動画が完成しました')).not.toBeInTheDocument();
  });

  it('links the verified artifact belonging to the completed job', () => {
    const artifacts: ProjectHistory['artifacts'] = [{ id: 10, job_id: 9, revision: 3, available: true, is_current: true,
      created_at: '2026-09-19T00:00:00Z', video_url: '/api/verified-video', subtitle_url: null }];
    card(generated(), historyFixture({ output_state: 'current', artifacts, jobs: [jobFixture({ id: 9, status: 'completed' })] }), 3);
    expect(screen.getByText('この要求の動画が完成しました')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'この要求で生成した動画を開く' })).toHaveAttribute('href', '/api/verified-video');
  });

  it.each(['failed', 'cancelled', 'unknown'])('shows %s generation with a recovery link', (status) => {
    card(generated(), historyFixture({ jobs: [jobFixture({ id: 9, status })] }), 3);
    expect(screen.getByRole('link', { name: '生成の履歴を確認する' })).toHaveAttribute('href', '#generation-history');
    expect(screen.queryByText('この要求の動画が完成しました')).not.toBeInTheDocument();
  });

  it.each([
    ['needs_input', '追加の内容を教えてください'], ['unsupported', 'この操作には対応していません'],
    ['blocked', '現在は実行できません'], ['error', '依頼を処理できませんでした'],
  ] as const)('keeps %s distinct from executed success', (status, heading) => {
    card(languageFixture('test', { status, result: null, executed: false }));
    expect(screen.getByRole('status')).toHaveTextContent(heading);
    expect(screen.getByText('この依頼による設定変更・動画生成は行っていません。')).toBeInTheDocument();
    expect(screen.queryByText('設定を保存しました')).not.toBeInTheDocument();
  });

  it('asks for a full edited request without pretending to retain a conversation', () => {
    card(languageFixture('test', { status: 'needs_input', result: null, executed: false,
      clarification: { kind: 'clarification', question: '何pxにしますか？', missing_fields: ['arguments'] } }));
    expect(screen.getByText('何pxにしますか？')).toBeInTheDocument();
    expect(screen.getByText('必要な内容を含めて、依頼全体を入力し直してください。')).toBeInTheDocument();
  });
});
