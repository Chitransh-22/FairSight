# ⚖️ FairSight — AI Bias Detection and Mitigation Dashboard

> Detect, visualize, and mitigate algorithmic bias in your datasets — powered by **AIF360**, **Reweighing**, and **local LLM-powered fairness explanations** via **Ollama**.

---

## 📌 Overview

FairSight is an interactive Streamlit dashboard that helps data scientists, ML engineers, and researchers audit their datasets for fairness issues. It automatically detects the dataset's likely business domain, measures **Disparate Impact**, **Statistical Parity Difference**, and **Consistency** across protected groups (e.g. gender, race, age), applies **Reweighing** mitigation via IBM's AIF360, and generates a plain-language fairness explanation using a **locally hosted LLM (Ollama)** — no external API keys or cloud calls required.

---

## 🗂 Project Structure

```
├── app.py                 # Streamlit entry point — orchestrates the full workflow
├── dataset_loader.py       # File upload, parsing (CSV / JSON), and validation
├── ui.py                   # Sidebar, dataset overview, configuration, and download UI
├── analysis_pipeline.py    # Orchestrates the end-to-end fairness analysis pipeline
├── bias_utils.py            # Core fairness utilities (validation, preprocessing, AIF360 wrappers)
├── models.py                # AnalysisResult dataclass — typed container for pipeline outputs
├── llm_utils.py              # Builds the fairness prompt and calls the local Ollama LLM
├── report_generator.py       # Assembles the final Markdown fairness report
├── visualization.py           # Plotly dashboard: KPI cards, comparison charts, outcome plots
├── requirements.txt
└── README.md
```

### `app.py`
The Streamlit entry point. Responsibilities include:
- Setting up the page config and title
- Rendering the sidebar and upload widget
- Managing session state (`df`, `analysis`, `ai_summary`)
- Triggering the fairness pipeline (`run_analysis`) when the user clicks **Analyze Bias**
- Rendering the dashboard, AI explanation, and download section
- Catching and surfacing pipeline errors via `st.error` / `st.exception`

### `dataset_loader.py`
Handles dataset upload and parsing:
- Accepts `.csv` and `.json` files
- Supports multiple JSON shapes: a top-level list, `{"data": [...]}`, `{"records": [...]}`, or arbitrary nested JSON (via `pd.json_normalize`)
- Validates that the uploaded file isn't empty and is properly encoded (UTF-8)
- Stores the parsed DataFrame in `st.session_state`

### `ui.py`
All non-dashboard UI components:
- `render_sidebar()` — branding, tech stack, supported formats, and workflow summary
- `render_dataset_overview(df)` — row/column counts, missing values, dtype breakdown, and a full data preview/inspection expander
- `render_configuration(df)` — target column and protected attribute selectors, plus the **Analyze Bias** button
- `render_ai_summary(ai_summary)` — displays the generated AI fairness explanation
- `render_downloads(analysis, ai_summary)` — download buttons for the mitigated dataset (CSV), metrics (JSON), and the full AI report (Markdown)

### `analysis_pipeline.py`
Coordinates the full fairness workflow via `run_analysis(df, target_col, protected_col)`:
1. Validates the dataset, target column, and protected attribute
2. Preprocesses and encodes the data
3. Detects the dataset's business domain/context
4. Resolves privileged/unprivileged group info
5. Wraps the data into an AIF360 `BinaryLabelDataset`
6. Measures fairness metrics **before** mitigation
7. Computes group outcome rates **before** mitigation
8. Applies **Reweighing** mitigation
9. Measures fairness metrics **after** mitigation
10. Computes weighted group outcome rates **after** mitigation
11. Returns everything bundled in an `AnalysisResult`

### `bias_utils.py`
Core fairness utility module wrapping pandas, scikit-learn, and AIF360 logic:

| Function | Description |
|---|---|
| `validate_dataset(df)` | Ensures the dataset exists, is a DataFrame, isn't empty, and has ≥2 columns |
| `validate_target(df, target_col)` | Ensures the target column exists, isn't fully missing, and has ≥2 classes |
| `validate_protected(df, protected_col)` | Ensures the protected attribute exists, isn't fully missing, and has ≥2 groups |
| `encode_target(series)` | Encodes the target into binary — handles already-binary, multi-class numeric (majority-vs-rest), and multi-class categorical columns |
| `encode_protected_attribute(series)` | Encodes the protected attribute into binary — handles binary, continuous numeric (median split), and multi-class categorical (majority-vs-others) columns |
| `encode_remaining_features(df, target_col, protected_col)` | Label-encodes remaining categorical feature columns, skipping obvious ID columns (`id`, `uuid`, `customer_id`, etc.) |
| `preprocess_data(df, target_col, protected_col)` | Full cleaning pipeline: drops duplicates/empty rows, imputes missing values (median/mode), and encodes target, protected, and remaining columns |
| `prepare_dataset(df, label_col, protected_col)` | Wraps a DataFrame into an AIF360 `BinaryLabelDataset` |
| `measure_bias(dataset, protected_col, priv, unpriv)` | Computes **Disparate Impact**, **Statistical Parity Difference**, and **Consistency** via `BinaryLabelDatasetMetric` |
| `mitigate_bias(dataset, protected_col, priv, unpriv)` | Applies the **Reweighing** algorithm from AIF360 |
| `group_outcome_rates(dataset, protected_col, use_weights=False)` | Returns per-group positive outcome rates, optionally weighted by instance weights (used after mitigation) |
| `get_group_info(df, protected_col)` | Resolves privileged/unprivileged values and human-readable labels for the protected attribute |
| `map_group_value(protected_col, value)` | Maps encoded values back to friendly labels (e.g. Male/Female, Majority/Others, Older/Younger) |
| `detect_dataset_context(df)` | Infers the likely business domain (Healthcare, Finance, HR, Education, Insurance, E-commerce, Marketing, Cybersecurity, Retail, Manufacturing, or General) from column-name keyword matching, with a confidence score |

### `models.py`
Defines `AnalysisResult`, the typed dataclass that carries every pipeline output — processed data, context, group info, both AIF360 datasets, and metrics/rates before and after mitigation — through the rest of the app.

### `llm_utils.py`
Generates the natural-language fairness explanation using a **local Ollama LLM** (default model: `gemma3:4b`, served at `http://127.0.0.1:11434`) via LangChain:
- Builds a structured prompt containing the domain, protected attribute, group labels, and all before/after metrics
- Instructs the model to explain the metrics, suggest possible causes, evaluate mitigation effectiveness, note remaining risks, and give recommendations — using only the supplied data, without inventing facts
- Returns the generated Markdown explanation to the app

### `report_generator.py`
Combines the deterministic analysis results and the LLM explanation into a single downloadable Markdown report (`build_full_report`), including dataset summary, before/after metrics, outcome rates, and the AI interpretation.

### `visualization.py`
Renders the Plotly-based fairness dashboard:
- `plot_kpi_cards` — DI, SPD, and Consistency metrics with before→after deltas
- `plot_metrics_comparison` — grouped bar chart comparing all three metrics before vs. after
- `plot_group_outcomes` — grouped bar chart of positive outcome rates by group, before vs. after
- `plot_bias_summary` — a simple success/warning banner indicating whether mitigation improved fairness

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fairsight.git
cd fairsight
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** AIF360 may require additional system dependencies. Refer to the [AIF360 installation guide](https://github.com/Trusted-AI/AIF360#installation) if you run into issues.

### 3. Set up Ollama (local LLM)

FairSight uses a **locally hosted LLM via [Ollama](https://ollama.com)** for AI-generated explanations — no API key required.

```bash
# Install Ollama, then pull the model used by the app
ollama pull gemma3:4b

# Make sure the Ollama server is running
ollama serve
```

By default, the app connects to `http://127.0.0.1:11434`. Update the `base_url` and `model` in `llm_utils.py` if you're running Ollama elsewhere or prefer a different model.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 🧭 How to Use

**Step 1 — Upload Dataset**
Upload a `.csv` or `.json` file. Common test datasets include Adult Income, COMPAS Recidivism, and German Credit.

**Step 2 — Review the Dataset Overview**
Inspect row/column counts, missing values, data types, and a preview of the raw data.

**Step 3 — Configure Analysis**
- Select the **Target Column** (the outcome to predict, e.g. `income`, `loan_approved`). Multi-class or non-binary targets are auto-converted to binary.
- Select the **Protected Attribute** (e.g. `gender`, `race`, `age`). Continuous or multi-class attributes are auto-converted to binary (median split or majority-vs-others).

**Step 4 — Run Analysis**
Click **🚀 Analyze Bias**. The pipeline will:
1. Validate and preprocess the data
2. Detect the dataset's likely business domain
3. Wrap the data into an AIF360 `BinaryLabelDataset`
4. Measure **Disparate Impact**, **Statistical Parity Difference**, and **Consistency** before mitigation
5. Apply **Reweighing** to rebalance instance weights
6. Re-measure all metrics after mitigation
7. Display the interactive dashboard and generate a local LLM fairness explanation

**Step 5 — Download Results**
Download the mitigated dataset (CSV), the raw metrics (JSON), and the full AI-generated report (Markdown).

---

## 📊 Key Concepts

### Disparate Impact (DI)
The ratio of positive outcome rates between the unprivileged and privileged groups:

```
DI = P(outcome=1 | unprivileged) / P(outcome=1 | privileged)
```

| DI Value | Interpretation |
|---|---|
| 0.8 – 1.25 | ✅ Fair (acceptable range) |
| 0.6 – 0.8 or 1.25 – 1.67 | ⚠️ Moderate bias |
| < 0.6 or > 1.67 | ❌ High bias |

A DI of **1.0** means both groups have equal positive outcome rates — perfect fairness.

### Statistical Parity Difference (SPD)
The absolute difference in positive outcome rates between the unprivileged and privileged groups. A value closer to **0** indicates better fairness.

### Consistency
Measures how similarly the model treats similar individuals (based on nearest neighbors). Higher values indicate more stable, individually fair outcomes.

### Reweighing (Mitigation)
A pre-processing technique from AIF360 that assigns different instance weights to compensate for bias in the training data. It does not alter the underlying data values — only the weights used when computing outcome rates and, in a downstream model, when training.

### Outcome Gap
The absolute difference in positive outcome rates between the privileged and unprivileged groups. A gap below **0.1** is generally considered acceptable.

---

## 🤖 AI Explanation

After analysis, FairSight sends the computed fairness metrics to a **local LLM served via Ollama** (default: `gemma3:4b`) to generate a plain-language explanation covering:
1. What the fairness metrics indicate
2. Possible causes of the observed bias
3. Whether Reweighing mitigation was effective
4. Remaining fairness risks
5. Practical recommendations
6. Limitations of the current analysis

The prompt explicitly instructs the model to reason **only from the supplied metrics** and to avoid inventing facts, since everything runs locally without external grounding.

---

## ⚠️ Limitations

- Only supports **binary classification** targets. Multi-class targets are auto-converted (majority-vs-rest or label encoding), which may lose meaning.
- Only supports **binary protected attributes**. Multi-valued or continuous attributes are collapsed to two groups (majority vs. others, or a median split).
- Mitigation uses only **Reweighing**. Other strategies (e.g. adversarial debiasing, post-processing) are not yet implemented.
- Domain detection (`detect_dataset_context`) is a keyword-based heuristic on column names — it is a best-effort classification, not a guarantee.
- Bias metrics reflect patterns in the **uploaded dataset** — they do not guarantee fairness of a deployed model on unseen data.
- The AI explanation depends on the quality and availability of the local Ollama model and should be reviewed critically.

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [AIF360](https://github.com/Trusted-AI/AIF360) | Fairness metrics and mitigation algorithms |
| [Scikit-learn](https://scikit-learn.org) | Label encoding utilities |
| [Pandas](https://pandas.pydata.org) | Data manipulation |
| [Plotly](https://plotly.com/python/) | Interactive dashboard charts |
| [Ollama](https://ollama.com) + [LangChain](https://www.langchain.com) | Local LLM-powered fairness explanations |

---

## 📄 License

This project is open-source. Feel free to use, modify, and distribute it with attribution.

---

## 🙋 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
