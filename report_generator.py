from analysis_pipeline import AnalysisResult
from llm_utils import format_rates

def build_full_report(analysis: AnalysisResult,
                      llm_text: str):
    
    label_map = analysis.group_info["mapping"]
    
    report = f"""# AI Fairness Report

---

Dataset Summary
--------------- 

**Domain:** {analysis.context["domain"]}

**Protected Attribute:** {analysis.group_info["protected_attribute"]}

**Privileged Group:** {analysis.group_info["privileged_label"]}

**Disadvantaged Group:** {analysis.group_info["unprivileged_label"]}

---

Fairness Metrics Before
-----------------------

- Disparate Impact: {analysis.metrics_before["disparate_impact"]:.3f}
- Statistical Parity Difference: {analysis.metrics_before["statistical_parity_difference"]:.3f}
- Consistency: {analysis.metrics_before["consistency"]:.3f}

---

Fairness Metrics After
----------------------

- Disparate Impact: {analysis.metrics_after["disparate_impact"]:.3f}
- Statistical Parity Difference: {analysis.metrics_after["statistical_parity_difference"]:.3f}
- Consistency: {analysis.metrics_after["consistency"]:.3f}

---

Outcome Analysis
----------------

### Before

{format_rates(
    analysis.rates_before, 
    label_map
)}

### After

{format_rates(
    analysis.rates_after, 
    label_map
)}

---

{llm_text}
"""

    return report