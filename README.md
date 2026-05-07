# A-Share ST Risk Predictor ⚠️

## 关于这个项目

这是我作为机器学习新手，通过 **Vibe Coding**（与 AI 协作编程）方式完成的第一个完整 ML 项目。

我负责：业务逻辑、数据决策、测试验证  
AI 负责：代码实现、调试、技术建议

整个项目从零开始，历时约 3 周，涵盖数据清洗、特征工程、建模调参、可解释性分析到 Streamlit Demo 的完整流程。对我来说，这是一次真正意义上的"从想法到产品"的实践。

---

## 项目背景

2024 年 4 月，证监会发布新"国九条"，大幅收紧上市公司监管，2023 年以来 ST 公司数量明显增加。

**核心想法**：能不能在公司被 ST **之前 1-2 年**，就从财务数据里发现风险信号？

这个项目的答案是：可以。模型在测试集上 Recall 达到 **87.3%**——即真正会被 ST 的公司中，有 87% 在 2 年前就被识别出来了。

---

## 有趣的发现

训练完模型后，我发现一件反直觉的事：

**审计意见（是否非标）的预测贡献，排在所有 31 个特征的最后两位。**

这其实很合理——审计师出具非标意见，往往是公司已经快 ST 的时候（T-1 年），而我们用的是 T-2 年数据。财务指标（资产增速、盈利能力）比审计意见**更早**反映出风险，这正是"提前预警"的价值所在。

---

---

# A-Share ST Risk Predictor ⚠️

> An interpretable machine learning system that predicts the risk of A-share listed companies being flagged as **ST / \*ST** (Special Treatment) **1–2 years in advance**.

## Key Features

- **Early Warning**: Uses T-2 year financial data to predict T-year ST events — not post-hoc diagnosis
- **Interpretable**: SHAP analysis reveals the top financial drivers behind each prediction
- **Interactive Demo**: Streamlit app with per-company risk scores and SHAP waterfall explanations
- **Reproducible**: Built on iFind offline exports — no paid API required at inference

---

## Methodology

### Sample Design

| | Description |
|---|---|
| **Positive samples** | 138 companies first flagged ST/\*ST in 2023–2025 |
| **Negative samples** | 1:1 matched controls — same SW industry, assets within ±30%, never ST'd |
| **Total dataset** | 276 companies × 30 financial features |

**Exclusions**: Financial sector (CSRC industry J), Beijing Stock Exchange, IPO < 3 years

### The T-2 Rule

| Predict | Input data |
|---------|-----------|
| 2023 ST risk | 2021 annual report |
| 2024 ST risk | 2022 annual report |
| 2025 ST risk | 2023 annual report |

Train on 2021+2022 cohorts → Test on 2023 cohort (time-based split, no data leakage)

---

## Results

| Model | AUC | F1 | Recall |
|-------|-----|----|--------|
| Logistic Regression | 0.701 | 0.649 | 0.676 |
| XGBoost (tuned) | 0.735 | — | — |
| **Random Forest ✅** | **0.791** | **0.747** | **0.873** |

**Recall of 87.3%** — the model correctly flags 87% of future ST companies two years early.

Note: Test metrics are computed on a 1:1 matched test set. Under the real A-share ST base rate (~1-2%), expected precision in production deployment would be substantially lower. A larger-scale evaluation with unmatched controls is left to future work.

---

## Top Risk Drivers (SHAP)

| Rank | Feature | Why it matters |
|------|---------|----------------|
| 1 | `total_assets_growth` | Rapid asset shrinkage is the strongest distress signal |
| 2 | `net_assets_growth` | Declining net assets → approaching insolvency |
| 3 | `eps` | Sustained losses directly trigger the ST threshold |
| 4 | `debt_to_asset` | High leverage amplifies bankruptcy risk |
| 5 | `net_margin` | Persistent negative margins precede ST events |

> **Notable finding**: Audit opinion features (`is_non_standard_audit`, `audit_opinion_code`) ranked last out of 31 features. Non-standard audit opinions typically appear at T-1 year — too late for early warning. Financial ratios detect risk earlier.

---

## Project Structure

```
A-Share-ST-Risk-Predictor/
├── data/
│   ├── raw/                              # iFind Excel exports
│   └── processed/
│       ├── 01_positive_samples.csv       # 138 cleaned ST companies
│       ├── 02_financials_long.csv        # All-A-share financials (5,105 co. × 3 yrs)
│       └── 03_modeling_dataset.csv       # Final 276-row modeling dataset
├── notebooks/
│   ├── 01_clean_positive_samples.ipynb   # ST sample cleaning pipeline
│   ├── 02_clean_financial_data.ipynb     # Financial data cleaning
│   ├── 03_match_and_merge.ipynb          # 1:1 matching + dataset assembly
│   ├── 04_modeling.ipynb                 # Training, tuning & evaluation
│   └── 05_shap_analysis.ipynb            # SHAP interpretability analysis
├── models/
│   ├── best_model.pkl                    # Trained Random Forest + preprocessors
│   ├── shap_summary.png
│   ├── shap_importance.png
│   └── shap_waterfall.png
├── app/
│   └── app.py                            # Streamlit demo
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/A-Share-ST-Risk-Predictor.git
cd A-Share-ST-Risk-Predictor
pip install -r requirements.txt
streamlit run app/app.py
```

> **Note**: Raw data files are not included due to size. Processed datasets and the trained model are provided directly.

---

## Tech Stack

| | Tool |
|--|------|
| Data | iFind Financial Terminal (offline Excel) |
| Processing | pandas, numpy |
| ML | scikit-learn, XGBoost |
| Interpretability | SHAP |
| Demo | Streamlit |
| Language | Python 3.12 |

---

## Limitations & Future Work

- **Sample size**: 276 samples is sufficient for proof-of-concept; a larger dataset would improve generalization
- **Forward prediction**: Downloading 2024 annual reports from iFind would enable genuine 2026 ST risk prediction
- **Text features**: MD&A or earnings call sentiment could add incremental signal

---

*Built as a portfolio project exploring applied ML for Chinese capital markets.*  
*First complete ML project — developed via Vibe Coding with AI assistance.*
