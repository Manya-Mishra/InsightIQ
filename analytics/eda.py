
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. OUTLIER DETECTION USING IQR
# --------------------------------------------------

def detect_outliers_iqr(df):

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    results = []

    for column in numeric_columns:

        values = df[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = values[
            (values < lower_bound) |
            (values > upper_bound)
        ]

        results.append({
            "Column": column,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outlier Count": len(outliers),
            "Outlier Percentage": round(
                (len(outliers) / len(values)) * 100, 2
            )
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 2. CORRELATION ANALYSIS
# --------------------------------------------------

def calculate_correlation(df):

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.shape[1] < 2:
        return None

    correlation_matrix = numeric_df.corr(
        method="pearson"
    )

    return correlation_matrix


# --------------------------------------------------
# 3. CATEGORICAL ANALYSIS
# --------------------------------------------------

def analyze_categorical_column(df, column):

    category_counts = (
        df[column]
        .astype("object")
        .fillna("(Missing)")
        .value_counts()
        .reset_index()
    )

    category_counts.columns = [
        "Category",
        "Count"
    ]

    category_counts["Percentage"] = (
        category_counts["Count"] /
        len(df) * 100
    ).round(2)

    return category_counts


# --------------------------------------------------
# 4. NUMERIC DISTRIBUTION SUMMARY
# --------------------------------------------------

def get_distribution_summary(df, column):

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    if values.empty:
        return None

    summary = {
        "Mean": round(values.mean(), 2),
        "Median": round(values.median(), 2),
        "Standard Deviation": round(values.std(), 2),
        "Minimum": round(values.min(), 2),
        "Maximum": round(values.max(), 2),
        "Skewness": round(values.skew(), 2),
    }

    return summary


# --------------------------------------------------
# 5. NUMERIC RELATIONSHIP ANALYSIS
# --------------------------------------------------

def get_relationship_data(df, x_column, y_column):

    relationship_df = df[
        [x_column, y_column]
    ].copy()

    relationship_df[x_column] = pd.to_numeric(
        relationship_df[x_column],
        errors="coerce"
    )

    relationship_df[y_column] = pd.to_numeric(
        relationship_df[y_column],
        errors="coerce"
    )

    relationship_df = relationship_df.dropna()

    return relationship_df


# --------------------------------------------------
# 6. TIME TREND ANALYSIS
# --------------------------------------------------

def get_time_trend(
    df,
    date_column,
    value_column,
    frequency="M"
):

    trend_df = df[
        [date_column, value_column]
    ].copy()

    trend_df[date_column] = pd.to_datetime(
        trend_df[date_column],
        errors="coerce"
    )

    trend_df[value_column] = pd.to_numeric(
        trend_df[value_column],
        errors="coerce"
    )

    trend_df = trend_df.dropna()

    if trend_df.empty:
        return pd.DataFrame()

    trend_df["Period"] = (
        trend_df[date_column]
        .dt.to_period(frequency)
        .astype(str)
    )

    result = (
        trend_df.groupby(
            "Period",
            as_index=False
        )[value_column]
        .sum()
    )

    return result