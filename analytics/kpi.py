import pandas as pd


# --------------------------------------------------
# 1. FIND POSSIBLE BUSINESS COLUMNS
# --------------------------------------------------

def find_column_by_keywords(df, keywords):

    for column in df.columns:

        column_lower = str(column).lower()

        for keyword in keywords:

            if keyword in column_lower:
                return column

    return None


# --------------------------------------------------
# 2. DETECT IMPORTANT COLUMNS
# --------------------------------------------------

def detect_business_columns(df):

    business_columns = {
        "revenue": find_column_by_keywords(
            df,
            [
                "revenue",
                "sales",
                "amount",
                "income",
                "turnover"
            ]
        ),

        "profit": find_column_by_keywords(
            df,
            [
                "profit",
                "net_profit",
                "gross_profit"
            ]
        ),

        "quantity": find_column_by_keywords(
            df,
            [
                "quantity",
                "qty",
                "units",
                "items"
            ]
        ),

        "customer": find_column_by_keywords(
            df,
            [
                "customer_id",
                "customer",
                "client_id",
                "client"
            ]
        ),

        "order": find_column_by_keywords(
            df,
            [
                "order_id",
                "order",
                "transaction_id",
                "invoice_id"
            ]
        ),

        "product": find_column_by_keywords(
            df,
            [
                "product",
                "product_name",
                "item",
                "item_name"
            ]
        ),

        "date": find_column_by_keywords(
            df,
            [
                "date",
                "time",
                "timestamp"
            ]
        )
    }

    return business_columns


# --------------------------------------------------
# 3. CALCULATE BASIC KPIs
# --------------------------------------------------

def calculate_kpis(df):

    business_columns = detect_business_columns(df)

    kpis = []

    # ----------------------------------------------
    # Revenue
    # ----------------------------------------------

    revenue_column = business_columns["revenue"]

    if revenue_column:

        revenue = pd.to_numeric(
            df[revenue_column],
            errors="coerce"
        ).sum()

        kpis.append({
            "KPI": "Total Revenue",
            "Value": revenue,
            "Unit": "currency",
            "Source Column": revenue_column
        })

    # ----------------------------------------------
    # Profit
    # ----------------------------------------------

    profit_column = business_columns["profit"]

    if profit_column:

        profit = pd.to_numeric(
            df[profit_column],
            errors="coerce"
        ).sum()

        kpis.append({
            "KPI": "Total Profit",
            "Value": profit,
            "Unit": "currency",
            "Source Column": profit_column
        })

    # ----------------------------------------------
    # Quantity
    # ----------------------------------------------

    quantity_column = business_columns["quantity"]

    if quantity_column:

        quantity = pd.to_numeric(
            df[quantity_column],
            errors="coerce"
        ).sum()

        kpis.append({
            "KPI": "Total Units Sold",
            "Value": quantity,
            "Unit": "number",
            "Source Column": quantity_column
        })

    # ----------------------------------------------
    # Orders
    # ----------------------------------------------

    order_column = business_columns["order"]

    if order_column:

        total_orders = df[order_column].nunique()

    else:

        total_orders = len(df)

    kpis.append({
        "KPI": "Total Orders",
        "Value": total_orders,
        "Unit": "number",
        "Source Column": order_column or "Rows"
    })

    # ----------------------------------------------
    # Customers
    # ----------------------------------------------

    customer_column = business_columns["customer"]

    if customer_column:

        total_customers = df[customer_column].nunique()

        kpis.append({
            "KPI": "Total Customers",
            "Value": total_customers,
            "Unit": "number",
            "Source Column": customer_column
        })

    # ----------------------------------------------
    # Average Order Value
    # ----------------------------------------------

    if revenue_column and total_orders > 0:

        average_order_value = revenue / total_orders

        kpis.append({
            "KPI": "Average Order Value",
            "Value": average_order_value,
            "Unit": "currency",
            "Source Column": revenue_column
        })

    # ----------------------------------------------
    # Profit Margin
    # ----------------------------------------------

    if revenue_column and profit_column and revenue != 0:

        profit_margin = (
            profit / revenue
        ) * 100

        kpis.append({
            "KPI": "Profit Margin",
            "Value": profit_margin,
            "Unit": "percentage",
            "Source Column": profit_column
        })

    return pd.DataFrame(kpis)


# --------------------------------------------------
# 4. KPI SUMMARY
# --------------------------------------------------

def get_kpi_summary(df):

    kpi_df = calculate_kpis(df)

    return kpi_df