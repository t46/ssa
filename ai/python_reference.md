# Development Guidelines
## Core Development Rules
1. Package Management  
   - **uv** のみ使用、`pip` 禁止  
2. Code Quality  
   - **型ヒント必須**、公開 API に docstring  
   - 関数は小さく、88 桁制限  
3. Testing  
   - `uv run pytest`、バグ修正には回帰テストを追加  
4. Code Style  
   - PEP 8 命名、f-strings、早期 return

## Python Tools
- Format: `uv run ruff format .`  
- Lint  : `uv run ruff check . --fix`  
- Type  : `uv run pyright`

## Pull Requests
- 変更概要を簡潔に。ツール名や AI 利用は **PR に明記しない**。
