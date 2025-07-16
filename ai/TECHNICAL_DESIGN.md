# 社会科学論文自動執筆システム 技術設計書

## 1. システム概要

本システムは、World Values Survey (WVS) データを活用し、OpenAI Agents SDKを用いて社会科学論文を自動生成するパイプラインです。

## 2. プロジェクト構造

```
social-science-automation/
├── src/
│   ├── agents/              # OpenAI Agentsの実装
│   │   ├── ... # 各エージェントのロジック
│   │
│   ├── core/                # コアロジック
│   │   ├── __init__.py
│   │   ├── workflow.py           # エージェント間の調整
│   │   ├── ...
│   │
│   ├── tools/               # エージェント用ツール
│   │   ├── __init__.py
│   │   ├── pdf_parser.py         # (例)PDF質問票解析
│   │   ├── ...
│   │
│   └── utils/               # ユーティリティ
│       ├── __init__.py
│       ├── validators.py         # データ検証
│       └── logger.py            # ロギング
│
├── data/                    # データディレクトリ
│   ├── raw/                     # 生データ（DVC管理）
│   ├── processed/               # 処理済みデータ
│   └── outputs/                 # 分析結果
│
├── templates/               # テンプレート
│   ├── prompts/                 # LLMプロンプト
│   └── paper/                   # 論文テンプレート
│
├── tests/                   # テストコード
├── notebooks/               # 開発用ノートブック
└── docs/                    # ドキュメント
```

## 3. 技術スタック

### 基盤技術
- **Python 3.12+**
- **uv**: パッケージ管理
- **OpenAI Agents SDK**: エージェントワークフロー
