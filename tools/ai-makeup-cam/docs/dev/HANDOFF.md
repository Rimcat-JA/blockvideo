# 開発の引き継ぎメモ（クラウド → ローカル）

2026-11-05 時点。クラウドの Claude Code セッションで進めてきた開発を、ユーザーのローカル PC
（Windows、GPU は AMD RX 9070 XT）で続けるためのメモです。**ローカルの Claude は最初にこのメモを全部読んでください。**

---

## 1. ユーザーと進め方（必ず守ること）

- 応答は**日本語**で、**ステップごとに分かりやすく**説明する。ユーザーは主に Python と R を使い、ときどき VBA も使う。
- ユーザーの代名詞は **she / her / 彼女** のみ。
- 「**相談しながら**」進めたいという希望。仕様の判断が必要なところは選択肢を示して確認する。
- テスト素材は**女性の素材**を使う（ユーザーの希望）。
- ユーザーの GitHub アカウントは **Rimcat-JA**。push・PR はすべて `Rimcat-JA/blockvideo` に行う
  （`FidesTherese/blockvideo-upgrade-project` は書き込み権限がないので使わない）。
- 顔の写真や動画（ユーザー本人のもの、テスト素材）は**絶対にリポジトリに入れない**。

## 2. 目的（決まっていること）

就活のオンライン面接（Zoom / Teams / Google Meet）で、メイクをしなくても、**対面の面接でするしっかりメイクと同じ見た目**に見せたい。
Web カメラ映像にリアルタイムでメイク・補正をかけ、OBS の仮想カメラ（pyvirtualcam）として出力するアプリ
`tools/ai-makeup-cam/` を作っている。

- **対応する動き**＝日本の一般的なオンライン面接：座って話す、うなずく、歯を見せて笑う、少し首を振る・かしげる
  （左右 ±20°、上下 ±15°、傾き ±10°）、手元のメモを見る、顔より下での小さな手振り、**最初と最後のお辞儀**
  （顔が下がる・画面から外れる → 効果はふわっと消して戻す）。手を挙げる・長く顔を手で覆う・立ち上がるは対象外。
- **最優先は「自然さ」と「崩れないこと」**。迷ったら、効果を弱めるかフェードさせる（アーティファクトを出さない）。
- 面接中に**絶対に止まらない・固まらない**こと（エフェクトごとの例外は隔離済み）。
- ユーザーが望む追加：歯（ホワイトニング・色ムラなじませ。**形は変えない**）、シェーディング・ハイライト・涙袋、
  顔の形の補正（小顔・エラ・あご・目・小鼻・口角・唇・眉。控えめな上限、背景のゆがみ対策）、
  髪型（①アホ毛消し・シルエット＋「背景を撮影」→ ②いつもの髪型・前髪を重ねる → ③3Dで髪型ごと）、
  強めのメイクと強めの補正のプリセット。
- 将来：**案A**（実際のメイク写真からプリセットを自動生成）、**案B**（学習モデルによる仕上げ。
  3D モデル制作者に発注する本人の 3D モデルでペアデータを作り、A100/H100 で大きい先生モデル → 蒸留した
  小さい生徒モデルを ONNX Runtime + DirectML で RX 9070 XT 上で実行。強さスライダー付き、
  合格ライン評価（ちらつき、30 分あたりの崩れ回数、ディープフェイク判定の自己チェック）を通ったものだけ面接用）。

## 3. 現在の状態

### ブランチ

| ブランチ（Rimcat-JA/blockvideo） | 中身 |
|---|---|
| `claude/clever-mendel-cdh090` | **PR [Rimcat-JA/blockvideo#5](https://github.com/Rimcat-JA/blockvideo/pull/5)**。第1弾は完成・テスト 547 件成功。第2弾の**設定項目だけ**追加済み（エフェクト未登録なのでアプリの動作には影響なし） |
| `claude/wip-phase2` | 上記＋**作業途中のファイル**（下の表）。テストが一部失敗する。**続きはここから** |

### 第1弾（完成・PR 済み）

肌・リップ・チーク・アイシャドウ・アイライナー・眉・髪のエフェクト、顔追跡（One Euro フィルタ）、
髪と肌の検出（バックグラウンドスレッド）、パイプライン（例外の隔離、フェード、鏡像）、カメラ・仮想カメラ
（カメラが切れても送出を続ける）、PySide6 設定画面（config のメタデータからタブを自動生成）、
プリセット（natural=就活ナチュラル／bare=すっぴん風補正／polished=しっかりメイク／off）、CLI
（GUI／`--headless`／`--image IN --out OUT --compare`／`--list-cameras`／`--download-models`）、
起動スクリプト（`start_windows.bat` など）、日本語 README、ドキュメント
（`docs/3d-model-commission-spec.md`、`docs/capture-guide.md`）。
**実機（Web カメラ・OBS・Windows/macOS）では一度も動かしていない。**

### 第2弾（途中で停止）

| 項目 | 状態 | 次にやること |
|---|---|---|
| 設定項目（config.py: contour / tearbag / teeth / reshape / hairshape）、`FrameContext.background` | ✅ PR 済み | — |
| 歯 `effects/teeth.py` + `tests/test_effect_teeth.py` | 🟡 WIP ブランチ。テスト 26 件中 **2 件失敗**（`test_changes_stay_in_the_mouth_opening`, `test_close_up_faces_are_processed_at_reduced_resolution`） | 失敗を直す、CREMA-D の笑顔の動画で見た目と時間的な安定性を確認 |
| 立体感・涙袋 `effects/contour.py`（ContourEffect / TearbagEffect） | 🟡 WIP ブランチ。**テストなし**。config の contour の初期値を 0.15 → 0.35 に上げた途中の状態 | テストを書く、初期値の自然さを見直す（濃い肌でも灰色に浮かないか） |
| 顔の形 `effects/reshape.py` | ❌ 未着手 | 契約は `docs/dev/contract-phase2.md` の Special contracts |
| 髪型① `effects/hairshape.py` + `background.py`（背景を撮影） | ❌ 未着手 | 同上。pipeline / runner / gui への最小限の追加も |
| 統合（エフェクト登録、プリセット調整＋`strong.json`、プリセットの日本語表示名、13 タブの画面整理、高速化、README 更新） | ❌ 未着手 | `docs/dev/workflow-phase2.js` の Integrate の指示 |
| 動画での品質確認（`--video` CLI、女性 11 人の動画、お辞儀、数値評価）と修正 | ❌ 未着手 | 同 Video QA / Fix |
| 案A `makeup_cam/analyze.py` | ❌ 未着手（ファイルなし） | `docs/dev/workflow-plan-a.js` の指示（往復テストで精度を確認） |

各エージェントへの詳しい技術的な指示（手法・注意点）は `docs/dev/workflow-phase2.js` と
`docs/dev/workflow-plan-a.js` のプロンプトにそのまま残してある。設計の約束ごとは
`docs/dev/contract-phase1.md`・`docs/dev/contract-phase2.md`。

## 4. ローカルの開発環境

```bash
cd tools/ai-makeup-cam
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/fetch_test_media.py
```

- `fetch_test_media.py` は `.test-media/`（git 管理外）に、テスト用の静止画（NASA の宇宙飛行士の写真、
  パブリックドメイン）と、CREMA-D の**女性 11 人×4 本**の会話動画（ODbL）をダウンロードする。
- モデル（約 20 MB）は初回起動時に自動でダウンロードされる。テストでモデルを使うには
  `set AI_MAKEUP_CAM_MODELS_DIR=<モデルのフォルダ>`（未設定だとモデルを使うテストはスキップ）。
  `python -m makeup_cam --download-models` で取得でき、保存先は `%LOCALAPPDATA%\ai-makeup-cam\models`。
- テスト：`python -m pytest -q`（GUI テストは `QT_QPA_PLATFORM=offscreen` でも動く）。
- 見た目の確認用の画像・動画は `.test-media/renders/` など git 管理外に出す。
- ワークフロー（複数エージェント）を使う場合、ローカルでは CPU 数に応じて同時実行数が増える
  （`CLAUDE_CODE_WORKFLOW_MAX_CONCURRENT_AGENTS` で上書き可）。

## 5. 最初にやると良いこと（おすすめの順番）

1. **実機で動かす**：`start_windows.bat` で起動し、ユーザーの Web カメラ・OBS Virtual Camera・Zoom で確認。
   ここで見つかる問題（カメラ名・解像度・FPS・DirectShow/MSMF・OBS の仮想カメラ・高 DPI の画面など）を最優先で直す。
   注意：Windows ではカメラ名が取れず「カメラ N」になるため、入力に OBS Virtual Camera を選ぶと映像がループする
   （カメラ名の取得手段を追加したい）。
2. 第2弾の残り（歯の失敗テスト → 立体感のテスト → 顔の形 → 髪型① → 統合 → 動画での品質確認 → 修正）。
3. 案A（写真からプリセット）。完成後、CLI（`--from-photo` など）と GUI の「写真から作る」ボタンを追加。
4. 録画ツール（台本を画面に出す。`docs/capture-guide.md` §5.2）→ 案B。

## 6. 既知の問題・メモ（第1弾のレポートより）

- 速度：クラウドの 4 コアで、エフェクト合計 17〜19 ms、1 フレーム全体で約 41 ms（約 24 fps）。
  パイプラインに約 9 ms の余分（フレーム全体の uint8⇔float32 変換、鏡像、クリップ）→ ROI だけ変換するなどで高速化の余地。
- MediaPipe の Face Landmarker は最初のフレームの結果が縦に歪む（VIDEO モードで十数フレームで安定）。
  静止画処理（`--image`）は同じ画像を 20 フレーム流して回避している。
- 小さい顔・暗い顔は全体のフレームだと検出漏れしやすい → tracking.py は正方形の窓で検出している。
- アイシャドウは、まぶたが重めの人（テストの写真）では polished でも控えめに見える。髪の効果は初期値だとほぼ見えない。
- プリセットの表示名が英語のファイル名（natural など）のまま → 日本語の表示名に対応したい。
- macOS のカラーダイアログは、選んでいる途中のライブプレビューが効かない可能性がある。
- `requirements.txt` は `opencv-contrib-python`（MediaPipe が入れるものと同じ）。headless 版と混在させないこと。
- MediaPipe の C++ のログ（W0000…, XNNPACK）は消せないが無害。

## 7. プライバシー

ユーザー本人の顔の写真・動画・3D モデルは、悪用されるとディープフェイクを作れてしまう。
リポジトリ（公開）には絶対に入れず、暗号化したドライブに保存する（`docs/capture-guide.md` §7）。
vast.ai などで学習する場合は Secure Cloud を使い、終わったらすぐ削除する。
