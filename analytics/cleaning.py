import pandas as pd


def get_data_quality_report(df):
    """
    Analyze the quality of a DataFrame without changing it.
    """

    missing_values = df.isnull().sum()

    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_missing_values": int(missing_values.sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_with_missing_values": int(
            (missing_values > 0).sum()
        ),
    }

    return report


def get_missing_value_report(df):
    """
    Return missing-value counts and percentages for each column.
    """

    total_rows = len(df)

    report = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2),
    })

    if total_rows == 0:
        report["Missing Percentage"] = 0.0

    return report


def clean_dataset(
    df,
    remove_duplicates=True,
    trim_text=True,
    fill_missing=True,
    numeric_strategy="Median",
    categorical_strategy="Most frequent",
):
    """
    Clean a copy of the uploaded DataFrame.

    Returns:
        cleaned_df: cleaned copy of the dataset
        cleaning_log: list of changes made
    """

    cleaned_df = df.copy()
    cleaning_log = []

    # 1. Remove extra spaces from text values
    if trim_text:
        text_columns = cleaned_df.select_dtypes(
            include=["object", "string"]
        ).columns

        for column in text_columns:
            cleaned_df[column] = cleaned_df[column].map(
                lambda value: (
                    value.strip()
                    if isinstance(value, str)
                    else value
                )
            )

        cleaning_log.append(
            "Removed leading and trailing spaces from text values."
        )

    # 2. Remove duplicate rows
    if remove_duplicates:
        duplicates_before = int(
            cleaned_df.duplicated().sum()
        )

        cleaned_df = cleaned_df.drop_duplicates()

        cleaning_log.append(
            f"Removed {duplicates_before} duplicate rows."
        )

    # 3. Handle missing values
    if fill_missing:
        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns

        categorical_columns = cleaned_df.select_dtypes(
            include=["object", "string", "category"]
        ).columns

        # Numeric columns
        for column in numeric_columns:
            if cleaned_df[column].isnull().any():

                if numeric_strategy == "Median":
                    fill_value = cleaned_df[column].median()

                elif numeric_strategy == "Mean":
                    fill_value = cleaned_df[column].mean()

                elif numeric_strategy == "Zero":
                    fill_value = 0

                else:
                    continue

                if pd.notna(fill_value):
                    cleaned_df[column] = (
                        cleaned_df[column].fillna(fill_value)
                    )

                    cleaning_log.append(
                        f"Filled missing values in '{column}' "
                        f"using {numeric_strategy.lower()}."
                    )

        # Categorical columns
        for column in categorical_columns:
            if cleaned_df[column].isnull().any():

                if categorical_strategy == "Most frequent":
                    modes = cleaned_df[column].mode(
                        dropna=True
                    )

                    if not modes.empty:
                        fill_value = modes.iloc[0]
                    else:
                        fill_value = "Unknown"

                elif categorical_strategy == "Unknown":
                    fill_value = "Unknown"

                else:
                    continue

                cleaned_df[column] = (
                    cleaned_df[column].fillna(fill_value)
                )

                cleaning_log.append(
                    f"Filled missing values in '{column}' "
                    f"using '{fill_value}'."
                )

    return cleaned_df, cleaning_log
