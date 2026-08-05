import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.preprocessing import LabelEncoder


# Configuring the Mapping to help: map_group_values() and get_group_info()
GROUP_MAPPINGS = {
    "gender": {
        0: "Female",
        1: "Male"
    },
    "sex": {
        0: "Female",
        1: "Male"
    },
    "race": {
        0: "Others",
        1: "Majority"
    },
    "age": {
        0: "Younger",
        1: "Older"
    }
}

# =========================================================
#                     Validation
# =========================================================

def validate_dataset(df: pd.DataFrame):
    
    """
    Validate uploaded dataset.
    """

    if df is None:
        raise ValueError("Dataset is missing.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) < 2:
        raise ValueError(
            "Dataset must contain at least two columns."
        )

    return True

def validate_target(df: pd.DataFrame, target_col: str):
    
    """
    Validate target column.
    """

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found."
        )

    if df[target_col].isna().all():
        raise ValueError(
            f"Target column '{target_col}' contains only missing values."
        )

    if df[target_col].nunique(dropna=True) < 2:
        raise ValueError(
            "Target column must contain at least two classes."
        )

    return True

def validate_protected(df: pd.DataFrame, protected_col: str):
    
    """
    Validate protected attribute.
    """

    if protected_col not in df.columns:
        raise ValueError(
            f"Protected attribute '{protected_col}' not found."
        )

    if df[protected_col].isna().all():
        raise ValueError(
            f"Protected attribute '{protected_col}' contains only missing values."
        )

    if df[protected_col].nunique(dropna=True) < 2:
        raise ValueError(
            "Protected attribute must contain at least two distinct groups."
        )

    return True

# =========================================================
#                    Encode-Target
# =========================================================

def label_encode(series: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    """
    Label encode a pandas Series and return both the encoded
    series and the fitted LabelEncoder.
    """

    encoder = LabelEncoder()

    encoded = pd.Series(
        data=np.asarray(
            encoder.fit_transform(series.astype(str)),
            dtype=np.int64
        ),
        index=series.index,
        name=series.name
    )

    return encoded, encoder

def encode_target(series: pd.Series) -> pd.Series:
    """
    Convert the target column into a numeric binary/encoded series.
    """

    series = series.copy()

    # Already binary
    if series.nunique() == 2:

        if not pd.api.types.is_numeric_dtype(series):

            encoded, _ = label_encode(series)
            return encoded

        return series.astype(int)

    # Multi-class numeric
    if pd.api.types.is_numeric_dtype(series):

        majority = series.value_counts().idxmax()

        return (series == majority).astype(int)

    # Multi-class categorical

    encoded, _ = label_encode(series)
    return encoded
# =========================================================
#                    Encode-Protected
# =========================================================

def encode_protected_attribute(series: pd.Series) -> pd.Series:
    """
    Convert the protected attribute into a binary column.
    """

    series = series.copy()

    # --------------------------------------------------
    # Already Binary
    # --------------------------------------------------

    if series.nunique() == 2:

        if not pd.api.types.is_numeric_dtype(series):

            encoded, _ = label_encode(series)
            return encoded

        return series.astype(int)

    # --------------------------------------------------
    # Continuous Numeric Attribute
    # --------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        median = series.median()

        return (series >= median).astype(int)

    # --------------------------------------------------
    # Multi-Class Categorical Attribute
    # --------------------------------------------------

    majority = series.value_counts().idxmax()

    return (series == majority).astype(int)

# =========================================================
#                    Encode-Remaining-Features
# =========================================================

def encode_remaining_features(
    df: pd.DataFrame,
    target_col: str,
    protected_col: str
) -> pd.DataFrame:
    """
    Label encode all remaining categorical features.
    """

    df = df.copy()

    ID_KEYWORDS = {
        "id",
        "uuid",
        "guid",
        "customer_id",
        "employee_id",
        "patient_id",
        "record_id",
        "transaction_id",
        "user_id"
    }

    for col in df.columns:

        # ----------------------------------------------
        # Skip target and protected attribute
        # ----------------------------------------------

        if col in [target_col, protected_col]:
            continue

        # ----------------------------------------------
        # Skip obvious identifier columns
        # ----------------------------------------------

        column_name = col.lower()

        if any(keyword in column_name for keyword in ID_KEYWORDS):
            continue

        # ----------------------------------------------
        # Encode categorical columns
        # ----------------------------------------------

        if not pd.api.types.is_numeric_dtype(df[col]):

            encoded, _ = label_encode(df[col])
            df[col] = encoded

    return df

# =========================================================
#                    Preprocessing
# =========================================================

def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    protected_col: str
):
    """
    Clean dataset and convert it into a numerical DataFrame
    suitable for AIF360.
    """

    df = df.copy()

    # --------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------

    df.drop_duplicates(inplace=True)

    # --------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------

    df.dropna(
        how="all",
        inplace=True
    )

    # --------------------------------------------------
    # Fill missing values
    # --------------------------------------------------

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            df[col] = df[col].fillna(
                df[col].median()
            )

        else:

            mode = df[col].mode()

            if not mode.empty:

                df[col] = df[col].fillna(
                    mode.iloc[0]
                )

            else:

                df[col] = df[col].fillna(
                    "Unknown"
                )

    # --------------------------------------------------
    # Encode Target Attribute
    # --------------------------------------------------

    df[target_col] = encode_target(df[target_col])

    # --------------------------------------------------
    # Encode Protected Attribute
    # --------------------------------------------------

    df[protected_col] = encode_protected_attribute(df[protected_col])

    # --------------------------------------------------
    # Encode Remaining Categorical Columns
    # --------------------------------------------------

    df = encode_remaining_features(df, target_col, protected_col)

    return df

# =========================================================
#                       AIF360
# =========================================================

def prepare_dataset(df, label_col, protected_col):
    
    """
    Convert DataFrame into an AIF360 BinaryLabelDataset.
    """

    dataset = BinaryLabelDataset(
        df=df,
        label_names=[label_col],
        protected_attribute_names=[protected_col],
        favorable_label=1,
        unfavorable_label=0
    )

    return dataset

def measure_bias(dataset, protected_col, privileged_value, unprivileged_value):

    """
    Compute fairness metrics.
    """

    metric = BinaryLabelDatasetMetric(

        dataset,

        privileged_groups=[
            {protected_col: privileged_value}
        ],

        unprivileged_groups=[
            {protected_col: unprivileged_value}
        ]

    )

    results = {

        "disparate_impact":
            float(metric.disparate_impact()),

        "statistical_parity_difference":
            float(metric.statistical_parity_difference()),

        "consistency":
            float(np.asarray(metric.consistency()).item())

    }

    return results

def mitigate_bias(dataset, protected_col, privileged_value, unprivileged_value):

    """
    Apply Reweighing.
    """

    rw = Reweighing(

        privileged_groups=[
            {protected_col: privileged_value}
        ],

        unprivileged_groups=[
            {protected_col: unprivileged_value}
        ]

    )

    rw.fit(dataset)

    dataset_fixed = rw.transform(dataset)

    return dataset_fixed

# =========================================================
#                     Statistics
# =========================================================

def group_outcome_rates(dataset, protected_col, use_weights=False):
    
    """
    Calculate positive outcome rate for each protected group.
    """

    df = dataset.convert_to_dataframe()[0]

    label = dataset.label_names[0]

    if use_weights:

        df["weights"] = dataset.instance_weights

        grouped = (
            df.groupby(
                protected_col,
                group_keys=False
            ).apply(
                lambda x:
                (x[label] * x["weights"]).sum()
                / x["weights"].sum(),
                include_groups=False
            )
        )

    else:

        grouped = (df.groupby(protected_col)[label].mean())

    return grouped.to_dict()

def map_group_value(protected_col, value):
    
    """
    Convert encoded values into user-friendly labels.
    """

    col = protected_col.lower()

    if col in GROUP_MAPPINGS:

        return GROUP_MAPPINGS[col].get(value, str(value))

    return str(value)

def get_group_info(df, protected_col):
    """
    Determine privileged/unprivileged groups and return
    all group information required by the fairness pipeline.

    Returns
    -------
    dict
        {
            "protected_attribute":...,
            "privileged_value": ...,
            "unprivileged_value": ...,
            "privileged_label": ...,
            "unprivileged_label": ...,
            "mapping": {...}
        }
    """

    if protected_col not in df.columns:
        raise ValueError(
            f"Protected attribute '{protected_col}' not found."
        )

    values = sorted(df[protected_col].dropna().unique())

    if len(values) != 2:
        raise ValueError(
            "Protected attribute must contain exactly 2 groups after preprocessing."
        )

    # --------------------------------------------------
    # Binary values after preprocessing
    # --------------------------------------------------

    unprivileged_value = values[0]
    privileged_value = values[1]

    col = protected_col.lower()

    # --------------------------------------------------
    # Human-readable labels
    # --------------------------------------------------

    mapping = GROUP_MAPPINGS.get(
        col,
        {
            values[0]: f"{protected_col} Group 1",
            values[1]: f"{protected_col} Group 2"
        }
    )

    return {
        "protected_attribute": protected_col,

        "privileged_value": privileged_value,

        "unprivileged_value": unprivileged_value,

        "privileged_label": mapping.get(
            privileged_value,
            str(privileged_value)
        ),

        "unprivileged_label": mapping.get(
            unprivileged_value,
            str(unprivileged_value)
        ),

        "mapping": mapping
    }

# =========================================================
#                     AI Summary
# =========================================================

def detect_dataset_context(df):
    """
    Detect the likely business domain of the uploaded dataset
    based on its column names.

    Returns
    -------
    dict
        {
            "domain": "...",
            "confidence": "...",
            "reason": "...",
            "rows": ...,
            "columns": ...,
            "numeric_columns": ...,
            "categorical_columns": ...
        }
    """
    cols = [c.lower().strip() for c in df.columns]

    keywords = {
        "Healthcare": {

            "strong": [

                "patient",
                "diagnosis",
                "disease",
                "hospital",
                "medical",
                "medicine",
                "treatment",
                "cholesterol",
                "glucose",
                "blood_pressure",
                "blood_sugar",
                "heart_rate",
                "ecg",
                "bmi",
                "prescription",
                "symptom",
                "doctor",
                "clinic",
                "admission",
                "discharge"

            ],

            "weak": [

                "bp",
                "pulse",
                "height",
                "weight"

            ]
        },

        "Finance": {

            "strong": [

                "loan",
                "credit",
                "bank",
                "account",
                "mortgage",
                "transaction",
                "fraud",
                "interest",
                "credit_score",
                "default",
                "debt",
                "emi",
                "balance"

            ],

            "weak": [

                "salary",
                "income",
                "payment",
                "deposit",
                "withdrawal"

            ]
        },

        "Human Resources": {

            "strong": [

                "employee",
                "department",
                "promotion",
                "performance",
                "designation",
                "position",
                "manager",
                "company",
                "joining_date",
                "experience"

            ],

            "weak": [

                "education",
                "salary",
                "bonus",
                "leave",
                "attendance"

            ]
        },

        "Education": {

            "strong": [

                "student",
                "school",
                "college",
                "university",
                "exam",
                "marks",
                "grade",
                "cgpa",
                "gpa",
                "semester",
                "subject",
                "course"

            ],

            "weak": [

                "attendance",
                "assignment",
                "quiz",
                "teacher"

            ]
        },

        "Insurance": {

            "strong": [

                "policy",
                "claim",
                "premium",
                "coverage",
                "insurance",
                "insured",
                "deductible",
                "underwriting"

            ],

            "weak": [

                "risk",
                "accident",
                "vehicle"

            ]
        },

        "E-commerce": {

            "strong": [

                "order",
                "product",
                "seller",
                "cart",
                "checkout",
                "shipping",
                "delivery",
                "inventory",
                "sku",
                "purchase"

            ],

            "weak": [

                "customer",
                "price",
                "review",
                "rating",
                "discount"

            ]
        },

        "Marketing": {

            "strong": [

                "campaign",
                "advertisement",
                "conversion",
                "lead",
                "impression",
                "ctr",
                "click",
                "engagement"

            ],

            "weak": [

                "customer",
                "channel",
                "region",
                "segment"

            ]
        },

        "Cybersecurity": {

            "strong": [

                "malware",
                "phishing",
                "firewall",
                "attack",
                "threat",
                "intrusion",
                "vulnerability",
                "exploit",
                "ransomware",
                "payload"

            ],

            "weak": [

                "network",
                "packet",
                "ip",
                "port",
                "protocol"

            ]
        },

        "Retail": {

            "strong": [

                "store",
                "sales",
                "inventory",
                "supplier",
                "stock",
                "barcode",
                "warehouse",
                "retail"

            ],

            "weak": [

                "price",
                "discount",
                "customer"

            ]
        },

        "Manufacturing": {

            "strong": [

                "factory",
                "machine",
                "production",
                "equipment",
                "assembly",
                "quality",
                "maintenance",
                "sensor",
                "plant"

            ],

            "weak": [

                "temperature",
                "pressure",
                "rpm"

            ]
        }
    }

    IGNORE_COLUMNS = {

        "id",
        "uuid",
        "index",

        "age",
        "gender",
        "sex",
        "race",

        "label",
        "target",
        "class",

        "name",

        "date",
        "timestamp"
    }

    scores = {}

    for domain, keyword_sets in keywords.items():

        score = 0

        for col in cols:

            if col in IGNORE_COLUMNS:
                continue

            # Strong keywords
            if col in keyword_sets["strong"]:
                score += 3

            # Weak keywords
            elif col in keyword_sets["weak"]:
                score += 1

        scores[domain] = score


    # -------------------------------------------------
    # Find highest score
    # -------------------------------------------------

    highest_score = max(scores.values())

    best_domains = [

        domain

        for domain, score in scores.items()

        if score == highest_score

    ]


    # -------------------------------------------------
    # Determine final domain
    # -------------------------------------------------

    if highest_score == 0:

        best_domain = "General Tabular Dataset"

        confidence = "Unknown"

        matched = 0

    elif len(best_domains) > 1:

        best_domain = "General Tabular Dataset"

        confidence = "Low"

        matched = highest_score

    else:

        best_domain = best_domains[0]

        matched = highest_score

        if highest_score >= 9:

            confidence = "High"

        elif highest_score >= 4:

            confidence = "Medium"

        else:

            confidence = "Low"

    return {

    "domain": best_domain,

    "confidence": confidence,

    "score": highest_score,

    "rows": len(df),

    "columns": len(df.columns),

    "numeric_columns":
        len(df.select_dtypes(include="number").columns),

    "categorical_columns":
        len(df.select_dtypes(exclude="number").columns)

}

def get_verdict(metrics_before, metrics_after, rates_before, rates_after):

    di_before = metrics_before["disparate_impact"]
    di_after = metrics_after["disparate_impact"]

    spd_before = metrics_before["statistical_parity_difference"]
    spd_after = metrics_after["statistical_parity_difference"]

    gap_before = abs(
        max(rates_before.values()) - min(rates_before.values())
    )

    gap_after = abs(
        max(rates_after.values()) - min(rates_after.values())
    )

    EXCELLENT_DI = 0.95
    GOOD_DI = 0.80
    MODERATE_DI = 0.70
    LIMITED_DI = 0.50


    # DEFINING VERDICT VARIABLES
    fairness_decreased = (
        di_after < di_before
        or abs(spd_after) > abs(spd_before)
        or gap_after > gap_before
    )

    excellent = (
        di_after >= EXCELLENT_DI
        and abs(spd_after) <= 0.02
        and gap_after <= 0.02
    )

    good = (
        GOOD_DI <= di_after < EXCELLENT_DI
        and abs(spd_after) <= 0.05
        and gap_after <= 0.05
    )

    moderate = (
        MODERATE_DI <= di_after < GOOD_DI
        and abs(spd_after) <= 0.10
        and gap_after <= 0.10
    )

    improved = (
        di_after > di_before
        or abs(spd_after) < abs(spd_before)
        or gap_after < gap_before
    )

    limited_improvement = (
        improved
        and LIMITED_DI <= di_after < MODERATE_DI
        and abs(spd_after) <= 0.20
        and gap_after <= 0.20
    )

    # USING GUARD CLAUSE TECHNIQUE

    if fairness_decreased:
        return {
            "status": "decreased",
            "title": "🔴 Fairness Decreased",
            "message": (
                "Bias mitigation reduced overall fairness. "
                "Review the preprocessing and mitigation strategy."
            ),
            "color": "error"
        }

    if excellent:
        return {
            "status": "excellent",
            "title": "🟢 Excellent Fairness Achieved",
            "message": (
                "Bias mitigation was highly successful. "
                "Fairness metrics indicate near-perfect parity between groups."
            ),
            "color": "success"
        }

    if good:
        return {
            "status": "good",
            "title": "🟢 Good Fairness Achieved",
            "message": (
                "Fairness improved significantly. "
                "Minor disparities remain."
            ),
            "color": "success"
        }

    if moderate:
        return {
            "status": "moderate",
            "title": "🟡 Moderate Fairness",
            "message": (
                "Bias mitigation reduced disparities, "
                "but additional improvements are recommended."
            ),
            "color": "warning"
        }

    if limited_improvement:
        return {
            "status": "limited",
            "title": "🟠 Limited Improvement",
            "message": (
                "Fairness improved after mitigation, "
                "however noticeable bias still remains."
            ),
            "color": "warning"
        }

    return {
        "status": "poor",
        "title": "🔴 High Bias Detected",
        "message": (
            "Significant fairness issues remain. "
            "Additional mitigation techniques are recommended."
        ),
        "color": "error"
    }