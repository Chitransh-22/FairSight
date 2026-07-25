from dataclasses import dataclass
import pandas as pd
from aif360.datasets import BinaryLabelDataset

@dataclass
class AnalysisResult:

    processed_df: pd.DataFrame

    context: dict

    group_info: dict

    dataset: BinaryLabelDataset

    mitigated_dataset: BinaryLabelDataset

    metrics_before: dict

    metrics_after: dict

    rates_before: dict

    rates_after: dict