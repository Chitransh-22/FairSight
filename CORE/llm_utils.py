from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from CORE.models import AnalysisResult

llm = OllamaLLM(
    model="gemma3:4b",
    base_url="http://127.0.0.1:11434",
    temperature=0.2
)

output_parser = StrOutputParser()

chain = llm | output_parser

prompt_template = PromptTemplate.from_template("""
You are an AI Fairness Auditor.

Your task is to explain fairness metrics in a practical,
professional, and unbiased manner.

The protected attribute may represent ANY grouping variable
such as:

- Gender
- Race
- Age
- Education
- Occupation
- Department
- Region
- Credit History
- Customer Segment
- Any other categorical grouping.

Never assume anything beyond the provided information.

--------------------------------------------------

Dataset Domain:
{dataset_domain}

Protected Attribute:
{protected_col}

Privileged Group:
{privileged}

Disadvantaged Group:
{disadvantaged}

--------------------------------------------------

FAIRNESS METRICS

Before Mitigation

Disparate Impact (DI):
{before:.3f}

Statistical Parity Difference (SPD):
{spd_before:.3f}

Consistency:
{consistency_before:.3f}

Outcome Gap:
{gap_before:.3f}

Outcome Rates

{rates_before}

--------------------------------------------------

After Mitigation

Disparate Impact (DI):
{after:.3f}

Statistical Parity Difference (SPD):
{spd_after:.3f}

Consistency:
{consistency_after:.3f}

Outcome Gap:
{gap_after:.3f}

Outcome Rates

{rates_after}

--------------------------------------------------

OUTCOME RATES

Before

...

After

...

--------------------------------------------------

RULES

Interpret the supplied metrics exactly as follows:

• Disparate Impact (DI)
    - Closer to 1.0 = Better fairness
    - Moving toward 1.0 = Improvement
    - Moving away from 1.0 = Worsening

• Statistical Parity Difference (SPD)
    - Closer to 0 = Better fairness
    - Moving toward 0 = Improvement
    - Moving away from 0 = Worsening

• Consistency
    - Higher values indicate more stable predictions.
    - Lower values indicate less stable predictions.

When evaluating mitigation:

- Compare BEFORE and AFTER metrics.
- Clearly state whether fairness improved, worsened, or remained unchanged.
- Never claim fairness worsened if DI moved closer to 1 AND SPD moved closer to 0 unless there is strong contradictory evidence.
- If different metrics disagree, explain the disagreement instead of choosing one metric.

Use only the supplied information.

Never invent facts.

Never contradict the supplied metrics.

If evidence is insufficient, explicitly state that.

Before writing the report, verify:

✓ DI interpretation is correct.
✓ SPD interpretation is correct.
✓ Mitigation conclusion matches the supplied metrics.
✓ Do not contradict any metric.
""")

def format_rates(rates, label_map):

    lines = []

    for group, rate in rates.items():

        label = label_map.get(group, str(group))

        lines.append(
            f"- {label}: {rate:.2%}"
        )

    return "\n".join(lines)

def build_prompt(context, group_info, metrics_before, metrics_after, rates_before, rates_after):

    priv = group_info["privileged_value"]
    unpriv = group_info["unprivileged_value"]
    label_map = group_info["mapping"]

    gap_before = abs(rates_before[priv] - rates_before[unpriv])

    gap_after = abs(rates_after[priv] - rates_after[unpriv])

    return prompt_template.format(

        dataset_domain=context["domain"],

        protected_col=group_info["protected_attribute"],

        privileged=group_info["privileged_label"],

        disadvantaged=group_info["unprivileged_label"],

        before=metrics_before["disparate_impact"],
        after=metrics_after["disparate_impact"],

        spd_before=metrics_before["statistical_parity_difference"],
        spd_after=metrics_after["statistical_parity_difference"],

        consistency_before=metrics_before["consistency"],
        consistency_after=metrics_after["consistency"],

        gap_before=gap_before,
        gap_after=gap_after,

        rates_before=format_rates(
            rates_before,
            label_map
        ),

        rates_after=format_rates(
            rates_after,
            label_map
        )
    )

def get_ai_explanation(analysis: AnalysisResult):

    context = analysis.context
    group_info = analysis.group_info
    metrics_before = analysis.metrics_before
    metrics_after = analysis.metrics_after
    rates_before = analysis.rates_before
    rates_after = analysis.rates_after

    prompt = build_prompt(
        context, group_info,
        metrics_before, metrics_after,
        rates_before, rates_after)

    response = chain.invoke(prompt)

    return response