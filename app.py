
import hashlib
import io

import streamlit as st
import pandas as pd
import plotly.express as px
from pandas.errors import EmptyDataError

from analytics.cleaning import (
    get_data_quality_report,
    get_missing_value_report,
    clean_dataset,
)

from analytics.eda import (
    detect_outliers_iqr,
    calculate_correlation,
    analyze_categorical_column,
    get_distribution_summary,
    get_relationship_data,
    get_time_trend,
)


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Business Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Business Analytics Platform")

st.write(
    "Upload your dataset to generate analytics, KPIs, "
    "statistical summaries, visualizations, and insights."
)


# --------------------------------------------------
# 2. FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload a CSV or Excel dataset to begin.")
    st.stop()


# --------------------------------------------------
# 3. LOAD AND VALIDATE DATASET
# --------------------------------------------------

file_bytes = uploaded_file.getvalue()

# Identify the actual uploaded file, not just its filename.
file_key = hashlib.sha256(file_bytes).hexdigest()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))

    elif uploaded_file.name.lower().endswith(".xlsx"):
        df = pd.read_excel(
            io.BytesIO(file_bytes),
            engine="openpyxl"
        )

    else:
        st.error("Unsupported file format.")
        st.stop()

except EmptyDataError:
    st.error("The CSV file is empty.")
    st.stop()

except Exception as e:
    st.error(f"Unable to read the dataset: {e}")
    st.info(
        "Check that the file is valid, not corrupted, "
        "and in the selected format."
    )
    st.stop()


# --------------------------------------------------
# 4. DATASET VALIDATION
# --------------------------------------------------

if df.empty:
    st.error("The uploaded dataset contains no rows.")
    st.stop()

if len(df.columns) == 0:
    st.error("The dataset contains no columns.")
    st.stop()

# Standardize column names for analysis.
df.columns = df.columns.astype(str).str.strip()

if any(name == "" for name in df.columns):
    st.warning("Some columns have blank names.")

if df.columns.duplicated().any():
    st.warning(
        "Duplicate column names were detected. "
        "Rename them before proceeding with analysis."
    )
    st.dataframe(df.head(), use_container_width=True)
    st.stop()

st.success("Dataset uploaded and loaded successfully!")


# --------------------------------------------------
# 5. DATASET PREVIEW
# --------------------------------------------------

st.header("1. Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# --------------------------------------------------
# 6. DATA QUALITY OVERVIEW
# --------------------------------------------------

st.header("2. Dataset Overview")

quality = get_data_quality_report(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Rows", f"{quality['total_rows']:,}")

with col2:
    st.metric("Columns", f"{quality['total_columns']:,}")

with col3:
    st.metric(
        "Missing Values",
        f"{quality['total_missing_values']:,}"
    )

with col4:
    st.metric(
        "Duplicate Rows",
        f"{quality['duplicate_rows']:,}"
    )


# --------------------------------------------------
# 7. MISSING VALUE REPORT
# --------------------------------------------------

st.subheader("Missing Values by Column")

missing_report = get_missing_value_report(df)

st.dataframe(
    missing_report,
    use_container_width=True
)


# --------------------------------------------------
# 8. COLUMN INFORMATION
# --------------------------------------------------

st.header("3. Column Information")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

st.dataframe(
    column_info,
    use_container_width=True
)


# --------------------------------------------------
# 9. COLUMN CLASSIFICATION
# --------------------------------------------------

st.header("4. Column Classification")

numeric_columns = df.select_dtypes(
    include="number"
).columns.tolist()

datetime_columns = df.select_dtypes(
    include=["datetime", "datetimetz"]
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Numeric Columns**")
    st.write(numeric_columns or "None found")

with col2:
    st.write("**Categorical Columns**")
    st.write(categorical_columns or "None found")

with col3:
    st.write("**Datetime Columns**")
    st.write(datetime_columns or "None found")


# --------------------------------------------------
# 10. STATISTICAL SUMMARY
# --------------------------------------------------

st.header("5. Statistical Summary")

if numeric_columns:
    st.dataframe(
        df[numeric_columns].describe().T,
        use_container_width=True
    )
else:
    st.info("No numeric columns were found.")


# --------------------------------------------------
# 11. DATA VISUALIZATION
# --------------------------------------------------

st.header("6. Data Visualization")

if numeric_columns:

    chart_type = st.selectbox(
        "Choose a chart type",
        [
            "Histogram",
            "Box Plot",
            "Category-wise Bar Chart"
        ]
    )

    numeric_column = st.selectbox(
        "Select a numeric column",
        numeric_columns
    )

    if chart_type == "Histogram":

        fig = px.histogram(
            df,
            x=numeric_column,
            title=f"Distribution of {numeric_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart_type == "Box Plot":

        fig = px.box(
            df,
            y=numeric_column,
            title=f"Box Plot of {numeric_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    elif chart_type == "Category-wise Bar Chart":

        if categorical_columns:

            category_column = st.selectbox(
                "Select a categorical column",
                categorical_columns
            )

            chart_df = df[
                [category_column, numeric_column]
            ].copy()

            chart_df[category_column] = (
                chart_df[category_column]
                .astype("object")
                .fillna("(Missing)")
            )

            grouped_data = (
                chart_df.groupby(
                    category_column,
                    as_index=False,
                    observed=True
                )[numeric_column]
                .sum()
                .sort_values(
                    numeric_column,
                    ascending=False
                )
                .head(20)
            )

            if not grouped_data.empty:

                fig = px.bar(
                    grouped_data,
                    x=category_column,
                    y=numeric_column,
                    title=(
                        f"{numeric_column} by "
                        f"{category_column}"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info("No data is available for this chart.")

        else:
            st.info(
                "A category-wise bar chart requires "
                "at least one categorical column."
            )

else:
    st.info(
        "No numeric columns are available for "
        "numeric summaries or these charts."
    )


# --------------------------------------------------
# 12. DATA CLEANING CONTROLS
# --------------------------------------------------

st.divider()

st.header("7. Automated Data Cleaning")

st.write(
    "Select the operations you want to apply. "
    "Your original uploaded dataset will remain unchanged."
)

with st.expander("Configure Cleaning Options", expanded=True):

    remove_duplicates = st.checkbox(
        "Remove duplicate rows",
        value=True
    )

    trim_text = st.checkbox(
        "Remove leading and trailing spaces from text",
        value=True
    )

    fill_missing = st.checkbox(
        "Fill missing values",
        value=True
    )

    numeric_strategy = st.selectbox(
        "Numeric missing-value strategy",
        [
            "Median",
            "Mean",
            "Zero",
            "Leave unchanged"
        ]
    )

    categorical_strategy = st.selectbox(
        "Categorical missing-value strategy",
        [
            "Most frequent",
            "Unknown",
            "Leave unchanged"
        ]
    )

    st.caption(
        "Mean and median apply only to numeric columns. "
        "Choose strategies carefully because filling values "
        "can affect your analysis."
    )


# --------------------------------------------------
# 13. RUN CLEANING
# --------------------------------------------------

if st.button("Clean Dataset", type="primary"):

    with st.spinner("Cleaning your dataset..."):

        cleaned_df, cleaning_log = clean_dataset(
            df,
            remove_duplicates=remove_duplicates,
            trim_text=trim_text,
            fill_missing=fill_missing,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy
        )

        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["cleaning_log"] = cleaning_log
        st.session_state["cleaned_file_key"] = file_key

    st.success("Cleaning process completed successfully!")


# --------------------------------------------------
# 14. DISPLAY CLEANED DATASET
# --------------------------------------------------

cleaned_df = st.session_state.get("cleaned_df")

cleaned_file_key = st.session_state.get(
    "cleaned_file_key"
)

if (
    cleaned_df is not None
    and cleaned_file_key == file_key
):

    st.header("8. Cleaned Dataset")

    st.dataframe(
        cleaned_df.head(10),
        use_container_width=True
    )

    st.subheader("Cleaning Log")

    cleaning_log = st.session_state.get(
        "cleaning_log",
        []
    )

    if cleaning_log:

        for entry in cleaning_log:
            st.write(f"✅ {entry}")

    else:
        st.info(
            "No cleaning operations were selected."
        )


    # --------------------------------------------------
    # 15. BEFORE AND AFTER COMPARISON
    # --------------------------------------------------

    st.subheader("Before vs After Cleaning")

    original_quality = get_data_quality_report(df)

    cleaned_quality = get_data_quality_report(cleaned_df)

    before_col, after_col = st.columns(2)

    with before_col:

        st.markdown("### Original Dataset")

        st.metric(
            "Rows",
            f"{original_quality['total_rows']:,}"
        )

        st.metric(
            "Missing Values",
            f"{original_quality['total_missing_values']:,}"
        )

        st.metric(
            "Duplicate Rows",
            f"{original_quality['duplicate_rows']:,}"
        )

    with after_col:

        st.markdown("### Cleaned Dataset")

        st.metric(
            "Rows",
            f"{cleaned_quality['total_rows']:,}"
        )

        st.metric(
            "Missing Values",
            f"{cleaned_quality['total_missing_values']:,}"
        )

        st.metric(
            "Duplicate Rows",
            f"{cleaned_quality['duplicate_rows']:,}"
        )


    # --------------------------------------------------
    # 16. DOWNLOAD CLEANED DATASET
    # --------------------------------------------------

    st.subheader("Download Results")

    csv_data = cleaned_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="Download Cleaned Dataset (CSV)",
        data=csv_data,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )



# --------------------------------------------------
# 17. ADVANCED EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

st.divider()

st.header("9. Advanced Exploratory Data Analysis")

# Use cleaned data for analysis if cleaning has been
# performed on the currently uploaded file.
# Otherwise, use the original uploaded dataset.

if (
    st.session_state.get("cleaned_df") is not None
    and st.session_state.get("cleaned_file_key") == file_key
):
    analysis_df = st.session_state["cleaned_df"]

    st.info(
        "Advanced EDA is using your cleaned dataset."
    )

else:
    analysis_df = df

    st.info(
        "Advanced EDA is using your original dataset."
    )


# Identify columns from the selected analysis dataset.

eda_numeric_columns = analysis_df.select_dtypes(
    include="number"
).columns.tolist()

eda_categorical_columns = analysis_df.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

eda_datetime_columns = analysis_df.select_dtypes(
    include=["datetime", "datetimetz"]
).columns.tolist()


# --------------------------------------------------
# 17A. OUTLIER DETECTION
# --------------------------------------------------

st.subheader("A. Outlier Detection")

if eda_numeric_columns:

    outlier_report = detect_outliers_iqr(analysis_df)

    if not outlier_report.empty:

        st.write(
            "Potential outliers detected using the IQR method."
        )

        st.dataframe(
            outlier_report,
            use_container_width=True
        )

        selected_outlier_column = st.selectbox(
            "Select a column to visualize outliers",
            eda_numeric_columns,
            key="outlier_column"
        )

        fig = px.box(
            analysis_df,
            y=selected_outlier_column,
            title=f"Outlier Analysis: {selected_outlier_column}",
            points="outliers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("No numeric data available for outlier detection.")

else:
    st.info("No numeric columns found.")


# --------------------------------------------------
# 17B. CORRELATION ANALYSIS
# --------------------------------------------------

st.subheader("B. Correlation Analysis")

correlation_matrix = calculate_correlation(analysis_df)

if correlation_matrix is not None:

    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("Correlation Matrix")

    st.dataframe(
        correlation_matrix,
        use_container_width=True
    )

else:
    st.info(
        "At least two numeric columns are required "
        "for correlation analysis."
    )


# --------------------------------------------------
# 17C. DISTRIBUTION ANALYSIS
# --------------------------------------------------

st.subheader("C. Distribution Analysis")

if eda_numeric_columns:

    distribution_column = st.selectbox(
        "Select a numeric column",
        eda_numeric_columns,
        key="distribution_column"
    )

    distribution_summary = get_distribution_summary(
        analysis_df,
        distribution_column
    )

    if distribution_summary is not None:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Mean",
                f"{distribution_summary['Mean']:,.2f}"
            )

            st.metric(
                "Median",
                f"{distribution_summary['Median']:,.2f}"
            )

        with col2:
            st.metric(
                "Minimum",
                f"{distribution_summary['Minimum']:,.2f}"
            )

            st.metric(
                "Maximum",
                f"{distribution_summary['Maximum']:,.2f}"
            )

        with col3:
            st.metric(
                "Standard Deviation",
                f"{distribution_summary['Standard Deviation']:,.2f}"
            )

            st.metric(
                "Skewness",
                f"{distribution_summary['Skewness']:,.2f}"
            )

        distribution_chart = st.selectbox(
            "Choose distribution chart",
            ["Histogram", "Box Plot"],
            key="distribution_chart"
        )

        if distribution_chart == "Histogram":

            fig = px.histogram(
                analysis_df,
                x=distribution_column,
                title=f"Distribution of {distribution_column}",
                marginal="box"
            )

        else:

            fig = px.box(
                analysis_df,
                y=distribution_column,
                title=f"Box Plot of {distribution_column}"
            )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("No valid numeric values are available.")

else:
    st.info("No numeric columns found.")


# --------------------------------------------------
# 17D. CATEGORICAL ANALYSIS
# --------------------------------------------------

st.subheader("D. Categorical Analysis")

if eda_categorical_columns:

    selected_category = st.selectbox(
        "Select a categorical column",
        eda_categorical_columns,
        key="eda_category"
    )

    category_report = analyze_categorical_column(
        analysis_df,
        selected_category
    )

    st.dataframe(
        category_report.head(20),
        use_container_width=True
    )

    fig = px.bar(
        category_report.head(20),
        x="Category",
        y="Count",
        title=f"Category Distribution: {selected_category}",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.info("No categorical columns found.")


# --------------------------------------------------
# 17E. NUMERIC RELATIONSHIP ANALYSIS
# --------------------------------------------------

st.subheader("E. Relationship Analysis")

if len(eda_numeric_columns) >= 2:

    col1, col2 = st.columns(2)

    with col1:

        x_column = st.selectbox(
            "Select X-axis column",
            eda_numeric_columns,
            key="relationship_x"
        )

    with col2:

        y_column = st.selectbox(
            "Select Y-axis column",
            [
                column for column in eda_numeric_columns
                if column != x_column
            ],
            key="relationship_y"
        )

    relationship_df = get_relationship_data(
        analysis_df,
        x_column,
        y_column
    )

    if not relationship_df.empty:

        fig = px.scatter(
            relationship_df,
            x=x_column,
            y=y_column,
            title=f"{x_column} vs {y_column}",
            trendline="ols"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        correlation_value = relationship_df[
            [x_column, y_column]
        ].corr().iloc[0, 1]

        st.metric(
            "Pearson Correlation",
            f"{correlation_value:.3f}"
            if pd.notna(correlation_value)
            else "Not available"
        )

    else:
        st.info("No valid data available for this relationship.")

else:
    st.info(
        "At least two numeric columns are required "
        "for relationship analysis."
    )


# --------------------------------------------------
# 17F. TIME TREND ANALYSIS
# --------------------------------------------------

st.subheader("F. Time Trend Analysis")

if eda_datetime_columns and eda_numeric_columns:

    col1, col2 = st.columns(2)

    with col1:

        selected_date = st.selectbox(
            "Select a date column",
            eda_datetime_columns,
            key="trend_date"
        )

    with col2:

        selected_value = st.selectbox(
            "Select a numeric metric",
            eda_numeric_columns,
            key="trend_value"
        )

    frequency_option = st.selectbox(
        "Select time frequency",
        ["Daily", "Weekly", "Monthly", "Quarterly"],
        key="trend_frequency"
    )

    frequency_mapping = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q"
    }

    trend_data = get_time_trend(
        analysis_df,
        selected_date,
        selected_value,
        frequency=frequency_mapping[frequency_option]
    )

    if not trend_data.empty:

        fig = px.line(
            trend_data,
            x="Period",
            y=selected_value,
            markers=True,
            title=f"{selected_value} Trend Over Time"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            trend_data,
            use_container_width=True
        )

    else:
        st.info(
            "No valid date and numeric values were found."
        )

else:
    st.info(
        "Time trend analysis requires at least one "
        "datetime column and one numeric column. "
        "Date columns stored as text are not yet "
        "automatically detected."
    )


# --------------------------------------------------
# 18. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Current version: Dataset profiling, data cleaning, "
    "and advanced exploratory data analysis. "
    "Automated KPIs, business insights, and AI recommendations "
    "will be added in later milestones."
)
