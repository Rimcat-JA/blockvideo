/** UI states that used to mislead: empty number fields, YOLO reports, status labels. */
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { ProjectSettingsEditor } from '@/components/ProjectSettingsEditor';
import { LanguageYoloReport } from '@/components/LanguageYoloReport';
import { ProgressBar } from '@/components/ProgressBar';
import { languageFixture } from '@/test/language-fixtures';
import { projectFixture } from '@/test/project-fixtures';

describe('settings editor number fields', () => {
  it('does not save a cleared field as 0', () => {
    const onSave = vi.fn();
    render(<ProjectSettingsEditor project={projectFixture()} disabled={false} onSave={onSave} />);
    const speed = screen.getByLabelText('話す速さ');
    fireEvent.change(speed, { target: { value: '' } });
    expect(screen.getByText('数値の欄が空か、入力途中です。')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '設定を保存' })).toBeDisabled();
    fireEvent.change(speed, { target: { value: '1.2' } });
    fireEvent.click(screen.getByRole('button', { name: '設定を保存' }));
    expect(onSave).toHaveBeenCalledWith({ voicevox_speed_scale: 1.2 });
  });
});

describe('YOLO report', () => {
  it('does not claim auto-execution when nothing ran', () => {
    render(<LanguageYoloReport response={{
      ...languageFixture('yolo'), status: 'blocked', executed: false, execution_mode: 'yolo',
      yolo_report: { guessing_allowed: true, bypassed_guards: [], auto_confirmed: [], unresolved: null },
    }} />);
    expect(screen.getByText('YOLOモード：この依頼は実行しませんでした')).toBeInTheDocument();
    expect(screen.queryByText('確認なしで自動実行しました（YOLO）')).toBeNull();
  });
});

describe('progress labels', () => {
  it('shows Japanese stage and status names', () => {
    render(<ProgressBar progress={1} stage="done" status="completed" />);
    expect(screen.getByText('工程: 完了（完了）')).toBeInTheDocument();
  });
});
