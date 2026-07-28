# Mermaid 図レンダリング(オプション)

BlockVideo の `diagram` 型ブロックは、既定で **mermaid-cli (mmdc)** で本家 Mermaid 構文を忠実に描画します。
mmdc が無い環境でもパイプラインは止まらず、組み込みの PIL レンダラ(縦一列のみ)にフォールバックします。

## セットアップ

このフォルダで一度だけ:

```powershell
cd tools\mermaid
npm install
```

`@mermaid-js/mermaid-cli` と puppeteer が入り、puppeteer が同梱の Chromium をダウンロードします。

### システムの Chrome/Edge を使う(Chromium ダウンロードを省く)

`puppeteer-config.json` の `executablePath` をインストール済みの Chrome または Edge に書き換えてください。
Windows の Edge 例:

```json
{ "executablePath": "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe" }
```

この設定ファイルを mmdc に渡すには、環境変数で指定します:

```powershell
$env:MERMAID_PUPPETEER_CONFIG = "<path-to-repo>\tools\mermaid\puppeteer-config.json"
```

## mmdc のパス解決

`app/services/mermaid_renderer.py` は以下の順で mmdc を探します:

1. 設定 `mermaid_mmdc_path`(`.env` で `MERMAID_MMDC_PATH=...`)
2. 環境変数 `MERMAID_MMDC_PATH`
3. PATH 上の `mmdc`
4. なければ `npx --yes mmdc`

例(`tools/mermaid/node_modules/.bin/mmdc.cmd` を直接指定):

```
MERMAID_MMDC_PATH=<path-to-repo>\tools\mermaid\node_modules\.bin\mmdc.cmd
```

## セキュリティ

- mmdc は `asyncio.create_subprocess_exec` の **argv 配列** で起動(`shell=False`)。
- Mermaid ソースは一時 `.mmd` ファイル経由で渡し、シェルに展開しません。
- LLM 出力は Pydantic `VisualPlan` で検証済みのものだけ渡します(`diagram` 上限 20000 字)。
