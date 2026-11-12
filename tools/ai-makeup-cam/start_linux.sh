#!/usr/bin/env bash
# AI makeup cam launcher for Linux. First run creates .venv with python3.12 (else python3)
# and installs requirements.txt; later runs reinstall only when requirements.txt changed.
# Arguments are passed to the app. The virtual camera needs v4l2loopback (see README).

cd "$(dirname "$0")" || exit 1

fail() {
    echo
    echo "[エラー] $1"
    exit 1
}

if command -v python3.12 >/dev/null 2>&1; then
    PYTHON=python3.12
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    fail "Python 3 が見つかりません。パッケージマネージャーで python3 と python3-venv をインストールしてください。"
fi

if [ ! -x .venv/bin/python ]; then
    echo "[AI メイクカメラ] 初回セットアップ: Python の仮想環境 .venv を作成しています（$PYTHON）..."
    "$PYTHON" -m venv .venv || fail "仮想環境 .venv を作成できませんでした。"
fi

.venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
    fail ".venv の Python が古すぎます（3.10 以上が必要です）。.venv フォルダを削除し、Python 3.12 を入れてから起動し直してください。"

if ! cmp -s requirements.txt .venv/requirements.installed; then
    echo "[AI メイクカメラ] 必要なライブラリをインストールしています（初回は数分かかります）..."
    .venv/bin/python -m pip install --disable-pip-version-check --upgrade pip
    .venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt ||
        fail "ライブラリのインストールに失敗しました。インターネット接続を確認してください。「mediapipe」が見つからない場合は Python 3.12 を入れ、.venv フォルダを削除してから起動し直してください。"
    cp requirements.txt .venv/requirements.installed
fi

.venv/bin/python -m makeup_cam "$@" || fail "AI メイクカメラがエラーで終了しました。上のメッセージを確認してください。"
