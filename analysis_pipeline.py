from bias_utils import (validate_dataset, validate_target,
                        validate_protected, preprocess_data,
                        detect_dataset_context, get_group_info,
                        prepare_dataset, measure_bias, mitigate_bias, 
                        group_outcome_rates, get_verdict)
from models import AnalysisResult


def run_analysis(df,target_col,protected_col) -> AnalysisResult:
    """
    Executes the complete fairness analysis pipeline.

    Returns
    -------
    AnalysisResult
        Contains all intermediate and final results of the fairness analysis.
    """

    # -----------------------------
    # Validation
    # -----------------------------
    validate_dataset(df)

    validate_target(df, target_col)

    validate_protected(df, protected_col)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    processed_df = preprocess_data(
        df,
        target_col,
        protected_col
    )

    # -----------------------------
    # Dataset Context
    # -----------------------------
    context = detect_dataset_context(
        processed_df
    )

    # -----------------------------
    # Protected Groups
    # -----------------------------
    group_info = get_group_info(
        processed_df,
        protected_col
    )

    # -----------------------------
    # Fairness Dataset
    # -----------------------------
    dataset = prepare_dataset(
        processed_df,
        target_col,
        protected_col
    )

    # -----------------------------
    # Metrics Before
    # -----------------------------
    metrics_before = measure_bias(
        dataset,
        group_info["protected_attribute"],
        group_info["privileged_value"],
        group_info["unprivileged_value"]
    )

    # -----------------------------
    # Rates Before
    # -----------------------------
    rates_before = group_outcome_rates(
        dataset,
        protected_col
    )

    # -----------------------------
    # Mitigation
    # -----------------------------
    mitigated_dataset = mitigate_bias(
        dataset,
        protected_col,
        group_info["privileged_value"],
        group_info["unprivileged_value"]
    )

    # -----------------------------
    # Metrics After
    # -----------------------------
    metrics_after = measure_bias(
        mitigated_dataset,
        group_info["protected_attribute"],
        group_info["privileged_value"],
        group_info["unprivileged_value"]
    )

    # -----------------------------
    # Rates After
    # -----------------------------
    rates_after = group_outcome_rates(
        mitigated_dataset,
        protected_col,
        use_weights=True
    )

    verdict = get_verdict(
        metrics_before,
        metrics_after,
        rates_before,
        rates_after
    )

    return AnalysisResult(
    processed_df=processed_df,
    context=context,
    group_info=group_info,
    dataset=dataset,
    mitigated_dataset=mitigated_dataset,
    metrics_before=metrics_before,
    metrics_after=metrics_after,
    rates_before=rates_before,
    rates_after=rates_after,
    verdict=verdict
)