@echo off
rem AI makeup cam launcher for Windows (double-click).
rem First run: creates .venv with Python 3.12 (or "python"), installs requirements.txt.
rem Later runs reinstall only when requirements.txt changed. Arguments are passed to the app.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
set "STAMP=.venv\requirements.installed"

if not exist "%VENV_PY%" (
    echo [AI メイクカメラ] 初回セットアップ: Python の仮想環境 .venv を作成しています...
    py -3.12 -m venv .venv >nul 2>&1 || python -m venv .venv
    if not exist "%VENV_PY%" goto :no_python
)

"%VENV_PY%" -c "import sys; sys.exit(sys.version_info < (3, 10))" || goto :old_python

fc /b requirements.txt "%STAMP%" >nul 2>&1
if errorlevel 1 (
    echo [AI メイクカメラ] 必要なライブラリをインストールしています（初回は数分かかります）...
    "%VENV_PY%" -m pip install --disable-pip-version-check --upgrade pip
    "%VENV_PY%" -m pip install --disable-pip-version-check -r requirements.txt || goto :install_failed
    copy /y requirements.txt "%STAMP%" >nul
)

"%VENV_PY%" -m makeup_cam %*
if errorlevel 1 goto :app_failed
exit /b 0

:no_python
echo.
echo [エラー] Python が見つかりません。
echo   https://www.python.org/downloads/windows/ から Python 3.12 をインストールしてください。
echo   インストール時に「Add python.exe to PATH」にチェックを入れてください。
goto :fail

:old_python
echo.
echo [エラー] .venv の Python が古すぎます（3.10 以上が必要です）。
echo   .venv フォルダを削除し、Python 3.12 をインストールしてから、もう一度起動してください。
goto :fail

:install_failed
echo.
echo [エラー] ライブラリのインストールに失敗しました。
echo   インターネット接続を確認してください。「mediapipe」が見つからないというエラーの場合は、
echo   Python 3.12 をインストールし、.venv フォルダを削除してから、もう一度起動してください。
goto :fail

:app_failed
echo.
echo [エラー] AI メイクカメラがエラーで終了しました。上のメッセージを確認してください。

:fail
echo.
pause
exit /b 1
