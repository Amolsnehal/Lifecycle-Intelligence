import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Customer Engagement & Retention Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Engagement & Product Utilization Analytics")
st.caption("Retention strategy dashboard | Descriptive and diagnostic analytics")

DEFAULT_FILE = "European_Bank (1).csv"

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(DEFAULT_FILE)

    required = {
        "CustomerId", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    for col in ["HasCrCard", "IsActiveMember", "Exited"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "EstimatedSalary", "Year"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload bank customer CSV", type=["csv"])
    st.caption(f"If no file is uploaded, the app looks for `{DEFAULT_FILE}` in the same folder.")

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.info("Place the CSV beside app.py, or upload it using the sidebar.")
    st.stop()

# Data validation summary
binary_cols = ["HasCrCard", "IsActiveMember", "Exited"]
invalid_binary = {
    c: int((~df[c].dropna().isin([0, 1])).sum())
    for c in binary_cols
}
missing_cells = int(df.isna().sum().sum())
duplicate_rows = int(df.duplicated().sum())

# Create analytical fields
df["Engagement"] = np.where(df["IsActiveMember"] == 1, "Active", "Inactive")
df["ChurnStatus"] = np.where(df["Exited"] == 1, "Churned", "Retained")
df["CardStatus"] = np.where(df["HasCrCard"] == 1, "Has credit card", "No credit card")

balance_cutoff = df["Balance"].quantile(0.75)
df["BalanceTier"] = np.where(df["Balance"] >= balance_cutoff, "High balance (top 25%)", "Other balance")
df["ProductTier"] = np.where(df["NumOfProducts"] >= 2, "Multi-product", "Single-product")

def safe_rate(numerator, denominator):
    return float(numerator / denominator * 100) if denominator else 0.0

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    geographies = sorted(df["Geography"].dropna().unique().tolist())
    selected_geo = st.multiselect("Geography", geographies, default=geographies)
    engagement_options = ["Active", "Inactive"]
    selected_engagement = st.multiselect("Engagement", engagement_options, default=engagement_options)
    product_min = int(df["NumOfProducts"].min())
    product_max = int(df["NumOfProducts"].max())
    selected_products = st.slider(
        "Number of products",
        min_value=product_min,
        max_value=product_max,
        value=(product_min, product_max),
    )
    balance_min = float(df["Balance"].min())
    balance_max = float(df["Balance"].max())
    balance_range = st.slider(
        "Balance range",
        min_value=balance_min,
        max_value=balance_max,
        value=(balance_min, balance_max),
        step=max((balance_max - balance_min) / 100, 1.0),
        format="%.0f",
    )
    salary_min = float(df["EstimatedSalary"].min())
    salary_max = float(df["EstimatedSalary"].max())
    salary_range = st.slider(
        "Estimated salary range",
        min_value=salary_min,
        max_value=salary_max,
        value=(salary_min, salary_max),
        step=max((salary_max - salary_min) / 100, 1.0),
        format="%.0f",
    )
    churn_filter = st.selectbox("Churn status", ["All", "Retained", "Churned"])

filtered = df[
    df["Geography"].isin(selected_geo)
    & df["Engagement"].isin(selected_engagement)
    & df["NumOfProducts"].between(selected_products[0], selected_products[1])
    & df["Balance"].between(balance_range[0], balance_range[1])
    & df["EstimatedSalary"].between(salary_range[0], salary_range[1])
].copy()

if churn_filter != "All":
    filtered = filtered[filtered["ChurnStatus"] == churn_filter]

if filtered.empty:
    st.warning("No customers match these filters. Adjust the filters in the sidebar.")
    st.stop()

# Tabs
overview, products, premium, relationship, data_tab = st.tabs([
    "Overview", "Product Utilization", "High-Value Disengaged", "Relationship Strength", "Data Quality & Export"
])

with overview:
    st.subheader("Engagement and churn overview")
    total_customers = len(filtered)
    churn_rate = filtered["Exited"].mean() * 100
    retention_rate = 100 - churn_rate
    active_rate = safe_rate((filtered["IsActiveMember"] == 1).sum(), total_customers)
    avg_products = filtered["NumOfProducts"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers in view", f"{total_customers:,}")
    c2.metric("Churn rate", f"{churn_rate:.2f}%")
    c3.metric("Retention rate", f"{retention_rate:.2f}%")
    c4.metric("Active members", f"{active_rate:.2f}%")

    left, right = st.columns(2)
    engagement_churn = (
        filtered.groupby("Engagement", dropna=False)
        .agg(Customers=("CustomerId", "count"), Churn_Rate=("Exited", "mean"))
        .reset_index()
    )
    engagement_churn["Churn Rate (%)"] = engagement_churn["Churn_Rate"] * 100
    with left:
        fig = px.bar(
            engagement_churn, x="Engagement", y="Churn Rate (%)",
            text="Churn Rate (%)", title="Churn rate by engagement status",
            color="Engagement"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(yaxis_range=[0, max(10, engagement_churn["Churn Rate (%)"].max() * 1.2)])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        status_counts = filtered["ChurnStatus"].value_counts().rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(status_counts, names="Status", values="Customers", hole=0.45, title="Retained vs churned customers")
        st.plotly_chart(fig, use_container_width=True)

    # Engagement Retention Ratio: active retention rate / inactive retention rate
    rates = filtered.groupby("Engagement")["Exited"].apply(lambda x: 1 - x.mean())
    active_ret = float(rates.get("Active", np.nan))
    inactive_ret = float(rates.get("Inactive", np.nan))
    ratio = active_ret / inactive_ret if pd.notna(active_ret) and pd.notna(inactive_ret) and inactive_ret != 0 else np.nan
    st.markdown("**Engagement Retention Ratio**")
    st.write(
        f"Active retention: {active_ret*100:.2f}% | Inactive retention: {inactive_ret*100:.2f}% | "
        + (f"Ratio: {ratio:.2f}×" if pd.notna(ratio) else "Ratio unavailable for current filters")
    )
    st.caption("A ratio above 1 means the active group has a higher observed retention rate. This is an association, not proof of causation.")

with products:
    st.subheader("Product utilization and retention")
    product_summary = (
        filtered.groupby("NumOfProducts")
        .agg(
            Customers=("CustomerId", "count"),
            Churn_Rate=("Exited", "mean"),
            Avg_Balance=("Balance", "mean"),
            Active_Share=("IsActiveMember", "mean"),
        )
        .reset_index()
    )
    product_summary["Churn Rate (%)"] = product_summary["Churn_Rate"] * 100
    product_summary["Active Share (%)"] = product_summary["Active_Share"] * 100

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            product_summary, x="NumOfProducts", y="Churn Rate (%)",
            text="Churn Rate (%)", title="Churn rate by number of products"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            product_summary, x="NumOfProducts", y="Customers",
            text="Customers", title="Customer count by number of products"
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    card_summary = (
        filtered.groupby("CardStatus")
        .agg(Customers=("CustomerId", "count"), Churn_Rate=("Exited", "mean"))
        .reset_index()
    )
    card_summary["Churn Rate (%)"] = card_summary["Churn_Rate"] * 100
    fig = px.bar(
        card_summary, x="CardStatus", y="Churn Rate (%)",
        text="Churn Rate (%)", title="Credit-card ownership and churn"
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Product Depth Index (descriptive view)**")
    st.dataframe(product_summary[["NumOfProducts", "Customers", "Churn Rate (%)", "Active Share (%)", "Avg_Balance"]], use_container_width=True)

with premium:
    st.subheader("High-value disengaged customer detector")
    st.write("Find inactive customers above adjustable balance and salary thresholds. These are rule-based segments, not a validated prediction of future churn.")

    col1, col2 = st.columns(2)
    with col1:
        balance_threshold = st.number_input(
            "Minimum balance threshold",
            min_value=float(df["Balance"].min()),
            max_value=float(df["Balance"].max()),
            value=float(balance_cutoff),
            step=5000.0,
        )
    with col2:
        salary_threshold = st.number_input(
            "Minimum estimated salary threshold",
            min_value=float(df["EstimatedSalary"].min()),
            max_value=float(df["EstimatedSalary"].max()),
            value=float(df["EstimatedSalary"].quantile(0.75)),
            step=5000.0,
        )

    only_not_churned = st.checkbox("Show only currently retained customers (potential retention targets)", value=True)
    premium_customers = df[
        (df["IsActiveMember"] == 0)
        & (df["Balance"] >= balance_threshold)
        & (df["EstimatedSalary"] >= salary_threshold)
    ].copy()
    if only_not_churned:
        premium_customers = premium_customers[premium_customers["Exited"] == 0]

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Customers matching criteria", f"{len(premium_customers):,}")
    pc2.metric("Average balance", f"{premium_customers['Balance'].mean():,.2f}" if len(premium_customers) else "—")
    pc3.metric("Already churned", f"{int(premium_customers['Exited'].sum()):,}" if len(premium_customers) else "0")

    display_cols = [
        c for c in [
            "CustomerId", "Geography", "Age", "Balance", "EstimatedSalary",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"
        ] if c in premium_customers.columns
    ]
    st.dataframe(premium_customers[display_cols], use_container_width=True, hide_index=True)
    st.download_button(
        "Download matching customers as CSV",
        data=premium_customers[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="high_value_disengaged_customers.csv",
        mime="text/csv",
    )

    st.markdown("**High-Balance Disengagement Rate**")
    high_balance = df[df["Balance"] >= balance_threshold]
    high_balance_disengagement = safe_rate((high_balance["IsActiveMember"] == 0).sum(), len(high_balance))
    st.metric("Inactive share among high-balance customers", f"{high_balance_disengagement:.2f}%")
    st.caption("This KPI measures disengagement among high-balance customers, not the probability that they will churn.")

with relationship:
    st.subheader("Relationship strength scoring")
    st.write("An illustrative business rule score is used here. It is not a statistically validated loyalty score; adjust the weights after reviewing results with banking stakeholders.")

    scored = filtered.copy()
    scored["EngagementPoints"] = np.where(scored["IsActiveMember"] == 1, 2, 0)
    scored["ProductPoints"] = np.where(scored["NumOfProducts"] >= 2, 2, 0)
    scored["CardPoints"] = np.where(scored["HasCrCard"] == 1, 1, 0)
    scored["BalancePoints"] = np.where(scored["Balance"] >= balance_cutoff, 1, 0)
    scored["RelationshipStrengthScore"] = (
        scored["EngagementPoints"] + scored["ProductPoints"] +
        scored["CardPoints"] + scored["BalancePoints"]
    )
    scored["RelationshipTier"] = pd.cut(
        scored["RelationshipStrengthScore"],
        bins=[-1, 2, 4, 6],
        labels=["Weak", "Moderate", "Strong"]
    )

    tier_summary = (
        scored.groupby("RelationshipTier", observed=False)
        .agg(
            Customers=("CustomerId", "count"),
            Churn_Rate=("Exited", "mean"),
            Avg_Balance=("Balance", "mean"),
            Avg_Products=("NumOfProducts", "mean"),
            Active_Share=("IsActiveMember", "mean"),
        )
        .reset_index()
    )
    tier_summary["Churn Rate (%)"] = tier_summary["Churn_Rate"] * 100
    tier_summary["Retention Rate (%)"] = 100 - tier_summary["Churn Rate (%)"]
    tier_summary["Active Share (%)"] = tier_summary["Active_Share"] * 100

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            tier_summary, x="RelationshipTier", y="Customers",
            title="Customers by relationship-strength tier", text="Customers",
            category_orders={"RelationshipTier": ["Weak", "Moderate", "Strong"]}
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(
            tier_summary, x="RelationshipTier", y="Churn Rate (%)",
            title="Churn rate by relationship-strength tier", text="Churn Rate (%)",
            category_orders={"RelationshipTier": ["Weak", "Moderate", "Strong"]}
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        tier_summary[["RelationshipTier", "Customers", "Retention Rate (%)", "Churn Rate (%)", "Avg_Balance", "Avg_Products", "Active Share (%)"]],
        use_container_width=True,
        hide_index=True,
    )

with data_tab:
    st.subheader("Data quality and export")
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Missing cells", f"{missing_cells:,}")
    q3.metric("Duplicate rows", f"{duplicate_rows:,}")

    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

    if any(invalid_binary.values()):
        st.warning(f"Non-binary values found in binary fields: {invalid_binary}")
    else:
        st.success("Binary fields HasCrCard, IsActiveMember, and Exited contain only 0/1 values (excluding missing values).")

    st.markdown("**Filtered dataset preview**")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered dataset as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_bank_customers.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Interpretation note: this dataset contains an activity flag rather than detailed transaction/login histories. "
    "Findings describe associations in the supplied data and should not be interpreted as causal effects or guaranteed future churn."
)
