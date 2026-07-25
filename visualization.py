import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from models import AnalysisResult

def plot_kpi_cards(metrics_before, metrics_after):

    st.subheader("📊 Fairness Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Disparate Impact",
            f"{metrics_after['disparate_impact']:.3f}",
            delta=round(
                metrics_after["disparate_impact"]
                - metrics_before["disparate_impact"],
                3
            )
        )

    spd_before = metrics_before["statistical_parity_difference"]
    spd_after = metrics_after["statistical_parity_difference"]

    # Remove floating-point noise
    if abs(spd_before) < 1e-10:
        spd_before = 0.0

    if abs(spd_after) < 1e-10:
        spd_after = 0.0

    improvement = abs(spd_before) - abs(spd_after)

    with col2:

        st.metric(
            "Statistical Parity Difference",
            f"{spd_after:.3f}",
            delta=round(improvement, 3)
        )

    with col3:

        st.metric(
            "Consistency",
            f"{metrics_after['consistency']:.3f}",
            delta=round(
                metrics_after["consistency"]
                - metrics_before["consistency"],
                3
            )
        )

def plot_metrics_comparison(metrics_before, metrics_after):

    df = pd.DataFrame({
        "Metric":[
            "Disparate Impact",
            "SPD",
            "Consistency"
        ],

        "Before":[
            metrics_before["disparate_impact"],
            metrics_before["statistical_parity_difference"],
            metrics_before["consistency"]
        ],

        "After":[
            metrics_after["disparate_impact"],
            metrics_after["statistical_parity_difference"],
            metrics_after["consistency"]
        ]

    })

    fig = px.bar(
        df,
        x="Metric",
        y=["Before","After"],
        barmode="group",
        title="Fairness Metrics Comparison"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

def plot_group_outcomes(rates_before, rates_after):

    groups = list(rates_before.keys())

    df = pd.DataFrame({

        "Group":groups,

        "Before":[
            rates_before[g]
            for g in groups
        ],

        "After":[
            rates_after[g]
            for g in groups
        ]

    })

    fig = px.bar(
        df,
        x="Group",
        y=["Before","After"],
        barmode="group",
        title="Outcome Rates Before vs After"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

def plot_bias_summary(metrics_before, metrics_after):

    di_before = abs( 1 - metrics_before["disparate_impact"] )
    di_after = abs( 1 - metrics_after["disparate_impact"] )

    spd_before = metrics_before["statistical_parity_difference"]
    spd_after = metrics_after["statistical_parity_difference"]

    if (( di_after >= di_before ) and ( abs(spd_after) <= abs(spd_before) )):
        st.success("Bias mitigation improved fairness.")

    else:
        st.warning("Bias mitigation produced limited improvement.")


def plot_dashboard(analysis: AnalysisResult):

    metrics_before = analysis.metrics_before
    metrics_after = analysis.metrics_after
    rates_before = analysis.rates_before
    rates_after = analysis.rates_after

    st.header("📊 Fairness Dashboard")

    plot_kpi_cards(
        metrics_before,
        metrics_after
    )

    plot_metrics_comparison(
        metrics_before,
        metrics_after
    )

    plot_group_outcomes(
        rates_before,
        rates_after
    )

    plot_bias_summary(
        metrics_before,
        metrics_after
    )