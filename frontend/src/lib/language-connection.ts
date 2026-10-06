export interface LanguageConnection {
  status: 'unconfigured' | 'configured' | 'invalid_configuration' | 'listed' | 'model_missing' | 'unreachable' | 'invalid_response';
  model: string | null;
  base_url: string | null;
  cloud_fallback: false;
  operation_mode?: 'all_tools' | 'semantic' | 'stateful';
  /** Whether the server accepts unattended (YOLO) requests. */
  yolo_enabled?: boolean;
}

export const operationModeLabels = {
  all_tools: '検索なし（全操作から判断）',
  semantic: '意味検索',
  stateful: '状態付き検索',
} as const;

export const connectionMessages: Record<LanguageConnection['status'], string> = {
  unconfigured: '言葉で操作するAIが未設定です。通常の設定フォームは使えます。',
  configured: '接続はまだ確認していません。',
  invalid_configuration: 'AIの接続設定を確認してください。',
  listed: '接続先の一覧にモデルを確認しました。依頼を送ると実際の処理が始まります。',
  model_missing: '設定したモデルが接続先の一覧にありません。モデル名と読み込み状態を確認してください。',
  unreachable: '接続を確認できません。LM Studioの起動とサーバー設定を確認してください。',
  invalid_response: '接続先の応答を確認できません。サーバー設定を確認してください。',
};
