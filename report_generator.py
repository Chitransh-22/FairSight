from analysis_pipeline import AnalysisResult
from llm_utils import format_rates
from io import BytesIO

from reportlab.lib.pagesizes import letter

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib.enums import TA_CENTER

def build_full_report(analysis: AnalysisResult,
                      llm_text: str):
    
    label_map = analysis.group_info["mapping"]
    
    report = f"""# AI Fairness Report

---

## Dataset Summary

**Domain:** {analysis.context["domain"]}

**Protected Attribute:** {analysis.group_info["protected_attribute"]}

**Privileged Group:** {analysis.group_info["privileged_label"]}

**Disadvantaged Group:** {analysis.group_info["unprivileged_label"]}

---

## Fairness Metrics Before

- Disparate Impact: {analysis.metrics_before["disparate_impact"]:.3f}
- Statistical Parity Difference: {analysis.metrics_before["statistical_parity_difference"]:.3f}
- Consistency: {analysis.metrics_before["consistency"]:.3f}

---

## Fairness Metrics After

- Disparate Impact: {analysis.metrics_after["disparate_impact"]:.3f}
- Statistical Parity Difference: {analysis.metrics_after["statistical_parity_difference"]:.3f}
- Consistency: {analysis.metrics_after["consistency"]:.3f}

---

## Outcome Analysis

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


def generate_pdf_report(markdown_report: str) -> bytes:

    # creating IO Buffer
    pdf_buffer = BytesIO()

    # creating PDF Document
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
        title="FairSight Report"
    )

    # creating styles
    base_styles = getSampleStyleSheet()

    # modifying template/style
    custom_title_style = ParagraphStyle(
        'CustomTitle',
        parent=base_styles['Heading1'],
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        spaceAfter=15,
    )

    report_elements=[]

    report_elements.append(
        Paragraph(
            "AI Fairness Audit Report",
            custom_title_style
        )
    )

    report_elements.append(Spacer(1, 12))

    # 4. Convert markdown
    # Splits text by lines and processes headers, lists, and spacing
    lines = markdown_report.split('\n')

    for line in lines:

        stripped_line = line.strip()

        # Handle empty lines with a vertical spacer

        if not stripped_line:

            report_elements.append(Spacer(1, 6))

            continue
            

        # Match markdown syntax to ReportLab Paragraph components

        if stripped_line.startswith('# '):

            report_elements.append(Paragraph(stripped_line[2:], custom_title_style))

        elif stripped_line.startswith('## '):

            report_elements.append(Paragraph(stripped_line[3:], base_styles['Heading2']))

        elif stripped_line.startswith('### '):

            report_elements.append(Paragraph(stripped_line[4:], base_styles['Heading3']))

        elif stripped_line.startswith('* ') or stripped_line.startswith('- '):

            # Formats bullet items using ReportLab's HTML-like entity tag

            report_elements.append(Paragraph(f"&bull; {stripped_line[2:]}", base_styles['Normal']))

        else:

            report_elements.append(Paragraph(stripped_line, base_styles['Normal']))


    # 5. Build PDF
    try:
        doc.build(report_elements)
        pdf_buffer.seek(0)
        pdf_bytes = pdf_buffer.getvalue()

    except Exception as e:
        raise RuntimeError("Failed to generate PDF report.") from e
    
    finally:
        pdf_buffer.close()

    return pdf_bytes