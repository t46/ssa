### **WVS Wave 7（2017-2022）米国データを利用したパイプライン案**

* World Value Survey とは 1981年から7波にわたり100か国以上を対象とした大規模国際比較調査であり、政治・宗教・経済・幸福感など多岐にわたる価値観データを時系列で追跡できます。  
* 各国約1,000～1,500人の確率標本を用い、人口規模調整ウェイト（S017）と標準化ウェイト（S018／S019）が提供されるため、国内分析にも国際比較にも対応しやすい設計です。
* この研究では wave 7 (2017 - 2020) を対象にします

**ステップ 0\. 前提：必要ファイルをまとめて取得 (実行済み)**

https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp から下記データを取得

| 分類                   | ファイル名（v6.0.0 例）                                 | 目的                                 |
| :--------------------- | :------------------------------------------------------ | :----------------------------------- |
| **個票データ**         | `WVS_Cross-National_Wave_7_csv_v6_0.csv`  | 回答者レコード（66 国分一括）        |
| **英語マスター質問票** | `F00010738-WVS-7_Master_Questionnaire_2017-2020_English.pdf`    | 全質問文・選択肢・スキップパターン   |
| **データコードブック** | `F00011055-WVS7_Codebook_Variables_report_V6.0.pdf`                                   | 変数ラベル・欠測コード・ウェイト説明 |

**ステップ 1｜データ取り込み&前処理　(実行済み)**

1. 上記ファイルを `data/raw/` に自動ダウンロード。   
2. `WVS_Cross-National_Wave_7_csv_v6_0.csv` から `B_COUNTRY == 840（USA）`のレコードのみを抽出し `data/processed/usa_w7.csv` として保存。
3. 質問票 (`F00010738-WVS-7_Master_Questionnaire_2017-2020_English.pdf`) から質問を key とする JSON ファイルを生成し、`code-maps/questionnaire_map.json` として保存
4. コードブック (`F00011055-WVS7_Codebook_Variables_report_V6.0.pdf`) から変数を key とする JSON ファイルを生成し、`code-maps/codebook_map.json` として保存（全ての変数名は `meta-data/all_variable_names` に保存）

**ステップ 2｜セマンティック検索のための準備　(実行済み)**
1. クエリを投げると `codebook_map.json` の `labels` に対してセマンティック検索が走り、variable が返ってくる
2. クエリを投げるたびにマッピングを `spec/variables.yaml` に query: variable という形式で保存（例：幸福感:Q49, 政府信頼:Q57）。

**ステップ 3｜LLM による研究アイデア創出**

1. `F00010738-WVS-7_Master_Questionnaire_2017-2020_English.pdf` を読み、それを元に米国の人々の価値観に関して社会科学者が関心を持ちそうな研究テーマを3案、LLMに提案させる。目的／理論背景／検証仮説を含めさせる。
2. *目的→背景→仮説→分析指標* を YAML として spec/research.yaml に出力。


**ステップ 4｜分析コード生成と実行**

1. 分析コード src/analysis.py を生成：  
   * S017 ウェイト適用  
   * 欠測処理とカテゴリ再符号化  
   * ロジスティック／OLS／重回帰など仮説別モデル  
2. 結果（係数表・図表・考察）を outputs/ に保存。

**ステップ 5｜論文生成**

1. LLM が構造化テンプレート（IMRaD）に従い LaTeX でドラフト作成。   
2. ドラフトをブラッシュアップして最終的な論文を生成
3. LaTeX ファイルをコンパイルして pdf を生成