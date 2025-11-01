# Social Science Research Automation (SSA)

World Values Survey Wave 7 (2017-2022) の米国データを用いた社会科学研究の自動化パイプライン

## 概要

本プロジェクトは、World Values Survey (WVS) のデータを活用し、以下のプロセスを完全自動化します：
- データ前処理とセマンティック検索の準備
- LLMによる研究アイデアの創出
- 統計分析コードの自動生成と実行
- 学術論文の自動執筆とPDF生成

詳細な処理フローについては [`ai/social_science_research_automation.md`](ai/social_science_research_automation.md) を参照してください。

## セットアップ

### 1. 環境変数の設定

```bash
# 環境変数を設定（どちらか一方を選択）

# Anthropic Claude を使用する場合
export ANTHROPIC_API_KEY_SSA=your_anthropic_api_key
export LLM_PROVIDER=anthropic

# OpenAI GPT を使用する場合
export OPENAI_API_KEY_SSA=your_openai_api_key
export LLM_PROVIDER=openai

# または、.zshrc や .bashrc に追加して永続化
echo 'export ANTHROPIC_API_KEY_SSA=your_anthropic_api_key' >> ~/.zshrc
echo 'export LLM_PROVIDER=anthropic' >> ~/.zshrc
# または
echo 'export OPENAI_API_KEY_SSA=your_openai_api_key' >> ~/.zshrc
echo 'export LLM_PROVIDER=openai' >> ~/.zshrc

### 2. LLM設定のカスタマイズ

`config/llm_config.yaml` ファイルで、各LLMプロバイダーの詳細設定を変更できます：

```yaml
# デフォルトプロバイダー
default_provider: "anthropic"

# Anthropic設定
anthropic:
  model: "claude-sonnet-4-20250514"
  max_tokens: 4000
  temperature: 0.7

# OpenAI設定
openai:
  model: "gpt-4o"
  max_tokens: 4000
  temperature: 0.7
```

### 2. 依存関係のインストール

```bash
# 依存関係のインストール
uv sync

# 開発ツールの実行
uv run ruff format .  # フォーマット
uv run ruff check . --fix  # リント
uv run pyright  # 型チェック
```

## クイックスタート

### 1. データの準備

[Google Driveからデータをダウンロード](https://drive.google.com/file/d/1QmQXt5N6XBcXlYkNWd-5-Ocx6rqXvUdE/view?usp=drive_link)し、`data/`ディレクトリに配置してください。

### 2. パイプラインの実行

```bash
# パイプライン全体の実行
uv run run_pipeline.py
```

このコマンドで以下が自動実行されます：
1. WVSデータのダウンロードと前処理: TODO
2. セマンティック検索の準備: TODO
3. 研究テーマの自動生成
4. 統計分析の実行
5. 論文の自動執筆とPDF化

## プロジェクト構造

```
ssa/
├── config/
│   └── llm_config.yaml  # LLM設定ファイル
├── data/
│   ├── raw/          # 生データ（WVS公式サイトからダウンロード）
│   └── processed/    # 前処理済みデータ
├── code-maps/        # 質問票・コードブックのJSONマッピング
├── meta-data/        # 変数名などのメタデータ
├── spec/             # 研究仕様・変数マッピング
├── outputs/          # 分析結果・論文
├── ai/               # プロジェクト仕様書
└── src/              # ソースコード
    ├── generate_research_ideas.py    # 研究アイデア生成
    ├── generate_and_execute_analysis.py  # 分析コード生成・実行
    ├── generate_paper.py             # 論文生成
    ├── terminal_formatter.py         # ターミナル出力フォーマット
    └── tools/
        ├── search_papers.py          # 論文検索ツール
        └── semantic_search.py        # セマンティック検索ツール
```

## 使用データ

World Values Survey Wave 7 (2017-2022)
- 対象国：米国（B_COUNTRY = 840）
- サンプルサイズ：約1,500人
- 調査項目：政治・宗教・経済・幸福感など多岐にわたる価値観

## 技術スタック

- Python 3.11+
- uv (パッケージ管理)
- pytest (テスト)
- ruff (リンター/フォーマッター)
- pyright (型チェック)