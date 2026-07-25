from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models import AnalysisResult

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
○ Auto Detect
○ Ask User
○ General Dataset

Protected Attribute:
{protected_col}

Privileged Group:
{privileged}

Disadvantaged Group:
{disadvantaged}

Disparate Impact Before:
{before:.3f}

Disparate Impact After:
{after:.3f}

Outcome Gap:
{gap:.3f}

Outcome Rates Before:

{rates_before}

Outcome Rates After:

{rates_after}

--------------------------------------------------

FAIRNESS METRICS

Before

DI:
{before}

SPD:
{spd_before}

Consistency:
{consistency_before}

After

DI:
{after}

SPD:
{spd_after}

Consistency:
{consistency_after}

--------------------------------------------------

OUTCOME RATES

Before

...

After

...

--------------------------------------------------

RULES

DI closer to 1 indicates better fairness.

SPD closer to 0 indicates better fairness.

Higher consistency indicates more stable predictions.

Use only the supplied information.

Never invent facts.

Never contradict the supplied metrics.

If evidence is insufficient, explicitly state that.

--------------------------------------------------

Only generate the following Markdown sections:

## LLM Interpretation

Explain what the fairness metrics indicate.

## Possible Causes

Suggest likely reasons for the observed bias based ONLY on the supplied metrics.

## Effectiveness of Mitigation

Evaluate whether mitigation improved fairness.

## Risks

Discuss remaining fairness risks.

## Recommendations

Suggest practical improvements.

## Limitations

Mention any limitations of the current analysis.

Do not repeat the metric values.
Do not invent facts.
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

        gap=abs(
            rates_before[priv]
            - rates_before[unpriv]
        ),

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

    print("========== AI DEBUG ==========")

    print(type(context))
    print(type(group_info))
    print(type(metrics_before))
    print(type(metrics_after))
    print(type(rates_before))
    print(type(rates_after))

    prompt = build_prompt(
        context, group_info,
        metrics_before, metrics_after,
        rates_before, rates_after)
    
    print(len(prompt))

    print("Calling Ollama...")

    response = chain.invoke(prompt)

    print("========== RESPONSE ==========")
    print(repr(response))
    print("==============================")

    return response