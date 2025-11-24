import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "final_data_with_revenue_profit.csv"

st.set_page_config(
    page_title="Steeves & Associates – Revenue & Profitability Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main > div { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 5px; }
    .stAlert { margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Clear cache if requested
if st.session_state.get('clear_cache', False):
    st.cache_data.clear()
    st.session_state['clear_cache'] = False


# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================
@st.cache_data
def load_data(path: str = None, uploaded_file=None) -> pd.DataFrame:
    """Load and clean the sponsor data with enhanced error handling"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif path:
            df = pd.read_csv(path)
        else:
            return pd.DataFrame()
    except FileNotFoundError:
        st.error(f"❌ File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Debug info
    st.sidebar.info(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns")
    with st.sidebar.expander("🔍 View Available Columns"):
        st.write(list(df.columns))

    # Parse dates
    if "Worked Date" in df.columns:
        df["Worked Date"] = pd.to_datetime(df["Worked Date"], errors="coerce")
        if df["Worked Date"].dt.tz is not None:
            df["Worked Date"] = df["Worked Date"].dt.tz_localize(None)

        # Add time features
        df["Year"] = df["Worked Date"].dt.year
        df["Quarter"] = df["Worked Date"].dt.quarter
        df["Month"] = df["Worked Date"].dt.month
        df["Week"] = df["Worked Date"].dt.isocalendar().week
        df["YearMonth"] = df["Worked Date"].dt.to_period("M").dt.to_timestamp()
        df["YearQuarter"] = df["Worked Date"].dt.to_period("Q").astype(str)

    # Convert numeric columns
    numeric_cols = [
        "Cost", "Actual_Cost", "Billable Hours", "Billed Hours", "Non Billed Hours",
        "Hourly Billing Rate", "Extended Price", "Amount Billed",
        "Amount Deducted from Retainer", "Revenue", "Profit"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use existing Revenue or default to 0
    if "Revenue" not in df.columns:
        df["Revenue"] = 0.0

    # Use Actual_Cost as the primary Cost column
    if "Actual_Cost" in df.columns:
        df["Cost"] = df["Actual_Cost"]

    # Calculate Total Hours
    if "Billed Hours" in df.columns:
        df["Total Hours"] = df["Billed Hours"].fillna(0)

    # Calculate Profit
    if "Profit" not in df.columns:
        if "Cost" in df.columns:
            df["Profit"] = df["Revenue"] - df["Cost"]
        else:
            df["Profit"] = np.nan

    # Row-level margin
    df["Profit_Margin_Row"] = np.where(
        df["Revenue"] > 0,
        df["Profit"] / df["Revenue"],
        np.nan
    )

    # Realization rate
    if "Extended Price" in df.columns and "Revenue" in df.columns:
        df["Realization_Rate"] = np.where(
            df["Extended Price"] > 0,
            df["Revenue"] / df["Extended Price"],
            np.nan
        )

    return df


@st.cache_data
def get_data_summary(df: pd.DataFrame) -> dict:
    """Get high-level data summary statistics"""
    return {
        "total_records": len(df),
        "date_range": (df["Worked Date"].min(), df["Worked Date"].max()) if "Worked Date" in df.columns else (
        None, None),
        "unique_clients": df["Client Name"].nunique() if "Client Name" in df.columns else 0,
        "unique_projects": df["Project Name"].nunique() if "Project Name" in df.columns else 0,
        "unique_consultants": df["Resource Name"].nunique() if "Resource Name" in df.columns else 0,
        "total_revenue": df["Revenue"].sum() if "Revenue" in df.columns else 0,
        "total_cost": df["Cost"].sum() if "Cost" in df.columns else 0,
        "total_billable_hours": df["Billed Hours"].sum() if "Billed Hours" in df.columns else 0,
    }


def aggregate_profitability(df: pd.DataFrame, group_col: str, min_hours_filter: float = 0.0) -> pd.DataFrame:
    """Enhanced aggregation with additional metrics"""
    if group_col not in df.columns:
        st.error(f"Column '{group_col}' not found in dataframe")
        return pd.DataFrame()

    required = ["Revenue", "Cost", "Billed Hours"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return pd.DataFrame()

    # Build aggregation dictionary
    agg_dict = {
        "Revenue": ("Revenue", "sum"),
        "Cost": ("Cost", "sum"),
        "Billable_Hours": ("Billed Hours", "sum"),
        "Record_Count": ("Revenue", "count"),
    }

    # Add conditional aggregations
    if "Hourly Billing Rate" in df.columns:
        agg_dict["Avg_Billing_Rate"] = ("Hourly Billing Rate", "mean")
        agg_dict["Median_Billing_Rate"] = ("Hourly Billing Rate", "median")
    if "Client Name" in df.columns:
        agg_dict["Client"] = ("Client Name", "first")
    if "Project Category" in df.columns:
        agg_dict["Category"] = ("Project Category", "first")
    if "Status (Billable / Non-Billable)" in df.columns:
        agg_dict["Status"] = ("Status (Billable / Non-Billable)", "first")
    if "Realization_Rate" in df.columns:
        agg_dict["Avg_Realization_Rate"] = ("Realization_Rate", "mean")

    try:
        agg = df.groupby(group_col, dropna=False).agg(**agg_dict).reset_index()
    except Exception as e:
        st.error(f"❌ Error during aggregation: {str(e)}")
        return pd.DataFrame()

    if len(agg) == 0:
        return pd.DataFrame()

    # Force numeric columns
    for col in ["Revenue", "Cost", "Billable_Hours"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0)

    # Calculate derived metrics
    agg["Revenue_per_Hour"] = np.where(agg["Billable_Hours"] > 0, agg["Revenue"] / agg["Billable_Hours"], 0)
    agg["Cost_per_Hour"] = np.where(agg["Billable_Hours"] > 0, agg["Cost"] / agg["Billable_Hours"], 0)
    agg["Profit"] = agg["Revenue"] - agg["Cost"]
    agg["Profit_Margin"] = np.where(agg["Revenue"] > 0, agg["Profit"] / agg["Revenue"], 0)
    agg["Profit_per_Hour"] = np.where(agg["Billable_Hours"] > 0, agg["Profit"] / agg["Billable_Hours"], 0)
    agg["ROI"] = np.where(agg["Cost"] > 0, (agg["Profit"] / agg["Cost"]) * 100, 0)

    # Replace inf values
    agg = agg.replace([np.inf, -np.inf], 0)

    # Filter by minimum hours
    if min_hours_filter > 0:
        agg = agg[agg["Billable_Hours"] >= min_hours_filter].copy()

    return agg


def show_global_kpis(df: pd.DataFrame, show_comparison=False):
    """Enhanced KPI display with optional period comparison"""
    if df.empty:
        st.warning("⚠️ No records after applying filters.")
        return

    # Calculate metrics
    rev_series = pd.to_numeric(df.get("Revenue", 0), errors="coerce")
    cost_series = pd.to_numeric(df.get("Cost", 0), errors="coerce")
    hours_series = pd.to_numeric(df.get("Billed Hours", 0), errors="coerce")

    total_rev = rev_series.sum(skipna=True)
    total_cost = cost_series.sum(skipna=True)
    total_hours = hours_series.sum(skipna=True)
    profit = total_rev - total_cost
    rev_per_hour = total_rev / total_hours if total_hours > 0 else 0
    margin = (profit / total_rev * 100) if total_rev > 0 else 0

    if "Hourly Billing Rate" in df.columns:
        avg_rate = pd.to_numeric(df["Hourly Billing Rate"], errors="coerce").mean()
    else:
        avg_rate = 0

    # Display KPIs
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("💰 Total Revenue", f"${total_rev:,.0f}", help="Total revenue from all work in the current filter")
    col2.metric("💵 Total Profit", f"${profit:,.0f}", delta=f"{margin:.1f}% margin", help="Revenue minus costs")
    col3.metric("⏱️ Billed Hours", f"{total_hours:,.0f}", help="Total Billed Hours in the current filter")
    col4.metric("📊 Revenue/Hour", f"${rev_per_hour:.2f}", help="Average revenue per billed hour")
    col5.metric("💵 Avg Rate", f"${avg_rate:.2f}/hr" if avg_rate > 0 else "N/A", help="Average billing rate")


# ============================================================================
# SIDEBAR - FILE UPLOAD & FILTERS
# ============================================================================
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.title("🎯 Dashboard Controls")
st.sidebar.markdown("### 📁 Data Source")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Upload a CSV file with the same format as the template"
)

# Show expected format
with st.sidebar.expander("ℹ️ Expected CSV Format"):
    st.markdown("""
    **Required columns:**
    - Project Name, Client Name, Worked Date
    - Billed Hours, Revenue, Cost
    - Resource Name, Project Category, Hourly Billing Rate

    **Optional columns:**
    - Status (Billable / Non-Billable)
    - Task or Ticket Title, Extended Price, Actual_Cost
    """)

    # Sample template
    sample_data = {
        "Project Name": ["Sample Project 1", "Sample Project 2"],
        "Client Name": ["Client A", "Client B"],
        "Worked Date": ["2024-01-15", "2024-01-16"],
        "Billed Hours": [8.0, 6.5],
        "Revenue": [1200.0, 975.0],
        "Cost": [800.0, 650.0],
        "Resource Name": ["John Doe", "Jane Smith"],
        "Project Category": ["Consulting", "Development"],
        "Hourly Billing Rate": [150.0, 150.0],
        "Status (Billable / Non-Billable)": ["Billable", "Not Billable"],
        "Task or Ticket Title": ["Task 1", "Task 2"],
    }
    sample_df = pd.DataFrame(sample_data)

    st.download_button(
        "📥 Download CSV Template",
        sample_df.to_csv(index=False).encode("utf-8"),
        file_name="revenue_profitability_template.csv",
        mime="text/csv",
        help="Download a sample CSV file with the correct format"
    )

# Show loaded file
if uploaded_file is not None:
    st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
    if st.sidebar.button("🔄 Clear Cache & Reload"):
        st.cache_data.clear()
        st.rerun()
else:
    st.sidebar.info(f"📄 Using default: {DATA_PATH}")

st.sidebar.markdown("---")

# Load data
with st.spinner("Loading data..."):
    df_raw = load_data(uploaded_file=uploaded_file) if uploaded_file else load_data(path=DATA_PATH)

    if df_raw.empty:
        st.error("❌ Failed to load data or no data available. Please check the file format.")
        st.info("💡 Make sure your CSV file contains all required columns")
        st.stop()

    # Check required columns
    required_cols = ["Project Name", "Client Name", "Worked Date", "Billed Hours", "Revenue", "Cost"]
    missing_cols = [col for col in required_cols if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.info("💡 Please ensure your CSV file contains all required columns")
        with st.expander("Available columns in your file"):
            st.write(list(df_raw.columns))
        st.stop()

    data_summary = get_data_summary(df_raw)

# View mode selector
view_mode = st.sidebar.radio(
    "View Mode",
    ["📊 Projects / Clients", "👥 Consultants", "📈 Time Series", "🎯 Executive Summary"],
    index=0,
)
view_mode = view_mode.split(" ", 1)[1]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filters")

# Date filter
if "Worked Date" in df_raw.columns:
    min_date = df_raw["Worked Date"].min()
    max_date = df_raw["Worked Date"].max()

    date_preset = st.sidebar.selectbox(
        "Date Range Preset",
        ["Custom", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "Year to Date", "All Time"]
    )

    if date_preset == "Last 30 Days":
        filter_start_date = max_date - timedelta(days=30)
        filter_end_date = max_date
    elif date_preset == "Last 90 Days":
        filter_start_date = max_date - timedelta(days=90)
        filter_end_date = max_date
    elif date_preset == "Last 6 Months":
        filter_start_date = max_date - timedelta(days=180)
        filter_end_date = max_date
    elif date_preset == "Last Year":
        filter_start_date = max_date - timedelta(days=365)
        filter_end_date = max_date
    elif date_preset == "Year to Date":
        filter_start_date = pd.Timestamp(datetime(max_date.year, 1, 1))
        filter_end_date = max_date
    elif date_preset == "All Time":
        filter_start_date = min_date
        filter_end_date = max_date
    else:
        date_input_result = st.sidebar.date_input(
            "Custom Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_input_result, tuple) and len(date_input_result) == 2:
            filter_start_date, filter_end_date = date_input_result
        else:
            filter_start_date = min_date
            filter_end_date = max_date

    mask_date = (df_raw["Worked Date"] >= pd.to_datetime(filter_start_date)) & (
            df_raw["Worked Date"] <= pd.to_datetime(filter_end_date))
else:
    mask_date = np.ones(len(df_raw), dtype=bool)
    filter_start_date = None
    filter_end_date = None

# Status filter
status_mode = st.sidebar.selectbox(
    "Status Filter",
    ["All", "Billable only", "Not Billable only"],
    index=1,
)

if "Status (Billable / Non-Billable)" in df_raw.columns:
    if status_mode == "Billable only":
        mask_status = df_raw["Status (Billable / Non-Billable)"] == "Billable"
    elif status_mode == "Not Billable only":
        mask_status = df_raw["Status (Billable / Non-Billable)"] == "Not Billable"
    else:
        mask_status = np.ones(len(df_raw), dtype=bool)
else:
    mask_status = np.ones(len(df_raw), dtype=bool)

df_filtered = df_raw[mask_date & mask_status].copy()

# Handle Non Billable Hours
if status_mode == "Not Billable only" and "Non Billable Hours" in df_filtered.columns:
    df_filtered["Billed Hours"] = pd.to_numeric(df_filtered["Non Billable Hours"], errors="coerce").fillna(0)

st.sidebar.markdown("---")

# Additional filters
if "Project Category" in df_filtered.columns:
    proj_cats = sorted(df_filtered["Project Category"].dropna().unique().tolist())
    selected_proj_cats = st.sidebar.multiselect("Project Category", options=proj_cats, default=proj_cats)
    if selected_proj_cats:
        df_filtered = df_filtered[df_filtered["Project Category"].isin(selected_proj_cats)]

if "Client Name" in df_filtered.columns:
    clients = sorted(df_filtered["Client Name"].dropna().unique().tolist())
    selected_clients = st.sidebar.multiselect("Client", options=clients, default=clients)
    if selected_clients:
        df_filtered = df_filtered[df_filtered["Client Name"].isin(selected_clients)]

if "Status (Billable / Non-Billable)" in df_filtered.columns:
    statuses = sorted(df_filtered["Status (Billable / Non-Billable)"].dropna().unique().tolist())
    selected_status = st.sidebar.multiselect("Status", options=statuses, default=statuses)
    if selected_status:
        df_filtered = df_filtered[df_filtered["Status (Billable / Non-Billable)"].isin(selected_status)]

st.sidebar.markdown("---")

# Advanced filters
with st.sidebar.expander("⚙️ Advanced Filters"):
    min_hours = st.slider("Min total Billed Hours", 0.0, 100.0, 0.0, 0.5)
    st.caption(f"Filter out entities with less than {min_hours} total Billed Hours")

    if "Hourly Billing Rate" in df_filtered.columns:
        rate_values = df_filtered["Hourly Billing Rate"].dropna()
        if len(rate_values) > 0:
            rate_range = st.slider(
                "Hourly Rate Range ($)",
                float(rate_values.min()),
                float(rate_values.max()),
                (float(rate_values.min()), float(rate_values.max()))
            )
            df_filtered = df_filtered[
                (df_filtered["Hourly Billing Rate"].isna()) |
                ((df_filtered["Hourly Billing Rate"] >= rate_range[0]) &
                 (df_filtered["Hourly Billing Rate"] <= rate_range[1]))
                ]

# Profitability metric
profit_metric_choice = st.sidebar.selectbox(
    "Profitability Metric",
    ["Profit Margin (%)", "Profit per Hour"],
    index=0,
)

# Filter summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Filter Summary")
date_range_days = len(
    pd.date_range(filter_start_date, filter_end_date, freq='D')) if filter_start_date and filter_end_date else 0
st.sidebar.info(f"""
- **Records:** {len(df_filtered):,} / {len(df_raw):,}
- **Date Range:** {date_range_days} days
- **Clients:** {df_filtered['Client Name'].nunique() if 'Client Name' in df_filtered.columns else 0}
- **Projects:** {df_filtered['Project Name'].nunique() if 'Project Name' in df_filtered.columns else 0}
""")

# ============================================================================
# PAGE 1: PROJECTS / CLIENTS VIEW
# ============================================================================
if view_mode == "Projects / Clients":
    st.title("📊 Projects / Clients – Revenue & Profitability Analysis")
    st.markdown("*Analyze project and client performance metrics*")

    show_global_kpis(df_filtered)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        available_group_options = []
        if "Project Name" in df_filtered.columns:
            available_group_options.append("Project Name")
        if "Client Name" in df_filtered.columns:
            available_group_options.append("Client Name")
        if "Project Category" in df_filtered.columns:
            available_group_options.append("Project Category")

        if not available_group_options:
            st.error("❌ No valid grouping columns found in data")
            st.stop()

        group_level = st.selectbox("Group By", available_group_options, index=0,
                                   help="Choose the aggregation level for analysis")

    with col2:
        chart_type = st.selectbox("Chart Type", ["Scatter", "Bar", "Treemap"])

    if df_filtered.empty:
        st.warning("⚠️ No data available after applying filters.")
        st.info("💡 Try adjusting your date range or other filters.")
        st.stop()

    if group_level not in df_filtered.columns:
        st.error(f"❌ Column '{group_level}' not found in the data.")
        st.write("**Available columns:**", list(df_filtered.columns))
        st.stop()

    st.info(
        f"📊 Working with {len(df_filtered):,} records, {df_filtered[group_level].nunique()} unique {group_level}(s)")

    agg = aggregate_profitability(df_filtered, group_level, min_hours)

    if agg is None or len(agg) == 0 or agg.empty:
        st.warning("⚠️ No data available for the selected filters and grouping level.")
        st.info(f"💡 Try reducing the minimum hours filter (currently {min_hours}) or adjusting other filters.")
        st.stop()

    st.success(f"✅ Found {len(agg)} {group_level}(s) with at least {min_hours} Billed Hours")

    y_col = "Profit_Margin" if profit_metric_choice == "Profit Margin (%)" else "Profit_per_Hour"

    if chart_type == "Scatter":
        fig_scatter = px.scatter(
            agg, x="Revenue_per_Hour", y=y_col, size="Billable_Hours", hover_name=group_level,
            hover_data={"Revenue": ":,.0f", "Billable_Hours": ":,.1f", "Profit": ":,.0f",
                        "Profit_Margin": ":.1%", "Profit_per_Hour": ":.2f", "ROI": ":.1f"},
            color="Category" if "Category" in agg.columns else None,
            labels={"Revenue_per_Hour": "Revenue per Hour ($/hr)", "Profit_Margin": "Profit Margin",
                    "Profit_per_Hour": "Profit per Hour ($/hr)", "Category": "Project Category"},
            title=f"{group_level}: Revenue per Hour vs {profit_metric_choice}",
            template="plotly_white"
        )

        if y_col == "Profit_Margin":
            fig_scatter.update_yaxes(tickformat=".1%", title="Profit Margin")
        else:
            fig_scatter.update_yaxes(title="Profit per Hour ($/hr)")

        median_x = agg["Revenue_per_Hour"].median()
        median_y = agg[y_col].median()
        fig_scatter.add_hline(y=median_y, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.add_vline(x=median_x, line_dash="dash", line_color="gray", opacity=0.5)
        fig_scatter.update_layout(height=600, margin=dict(l=30, r=30, t=60, b=30))
        st.plotly_chart(fig_scatter, use_container_width=True)

        with st.expander("📖 Chart Interpretation Guide"):
            col1, col2 = st.columns(2)
            with col1:
                st.success("**Top-Right Quadrant**: High rate & high margin → Best performers")
                st.info("**Top-Left Quadrant**: Low rate but high margin → Pricing opportunity")
            with col2:
                st.warning("**Bottom-Right Quadrant**: High rate but low margin → Cost control needed")
                st.error("**Bottom-Left Quadrant**: Low rate & low margin → Lowest priority")

    elif chart_type == "Bar":
        agg_sorted = agg.sort_values("Revenue", ascending=True).tail(20)
        fig_bar = px.bar(agg_sorted, y=group_level, x="Revenue", orientation='h', color="Profit_Margin",
                         hover_data=["Profit", "Billable_Hours"], title=f"Top 20 {group_level} by Revenue",
                         template="plotly_white", color_continuous_scale="RdYlGn")
        fig_bar.update_layout(height=700)
        st.plotly_chart(fig_bar, use_container_width=True)

    else:  # Treemap
        fig_tree = px.treemap(agg.head(30), path=[group_level], values="Revenue", color="Profit_Margin",
                              hover_data=["Profit", "Billable_Hours"],
                              title=f"{group_level} Revenue Distribution (Top 30)",
                              color_continuous_scale="RdYlGn", template="plotly_white")
        fig_tree.update_layout(height=600)
        st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Top Performers")

    top_n = st.slider("Number of top performers to display", 5, 50, 10)
    col_a, col_b, col_c = st.columns(3)

    # Top by Revenue
    agg_sorted_rev = agg.sort_values("Revenue_per_Hour", ascending=False).head(top_n)
    fig_rev = px.bar(agg_sorted_rev, x=group_level, y="Revenue_per_Hour",
                     hover_data={"Revenue": ":,.0f", "Billable_Hours": ":,.1f", "Profit_Margin": ":.1%"},
                     labels={"Revenue_per_Hour": "$/hr"}, title=f"Top {top_n} by Revenue/Hour",
                     template="plotly_white", color="Revenue_per_Hour", color_continuous_scale="Blues")
    fig_rev.update_layout(xaxis_tickangle=-45, showlegend=False)
    col_a.plotly_chart(fig_rev, use_container_width=True)

    # Top by Profit
    agg_sorted_prof = agg.sort_values("Profit_per_Hour", ascending=False).head(top_n)
    fig_prof = px.bar(agg_sorted_prof, x=group_level, y="Profit_per_Hour",
                      hover_data={"Revenue": ":,.0f", "Billable_Hours": ":,.1f", "Profit_Margin": ":.1%"},
                      labels={"Profit_per_Hour": "$/hr"}, title=f"Top {top_n} by Profit/Hour",
                      template="plotly_white", color="Profit_per_Hour", color_continuous_scale="Greens")
    fig_prof.update_layout(xaxis_tickangle=-45, showlegend=False)
    col_b.plotly_chart(fig_prof, use_container_width=True)

    # Top by Margin
    agg_sorted_margin = agg.sort_values("Profit_Margin", ascending=False).head(top_n)
    fig_margin = px.bar(agg_sorted_margin, x=group_level, y="Profit_Margin",
                        hover_data={"Revenue": ":,.0f", "Profit": ":,.0f", "Billable_Hours": ":,.1f"},
                        labels={"Profit_Margin": "Margin"}, title=f"Top {top_n} by Profit Margin",
                        template="plotly_white", color="Profit_Margin", color_continuous_scale="RdYlGn")
    fig_margin.update_yaxes(tickformat=".1%")
    fig_margin.update_layout(xaxis_tickangle=-45, showlegend=False)
    col_c.plotly_chart(fig_margin, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Detailed Drill-Down")

    selected_group = st.selectbox(f"Select a {group_level} for detailed analysis",
                                  options=sorted(agg[group_level].dropna().unique().tolist()))

    detail_df = df_filtered[df_filtered[group_level] == selected_group].copy()

    if not detail_df.empty:
        st.write(f"#### 📋 {selected_group}")

        detail_agg = aggregate_profitability(detail_df, group_level, 0)
        if not detail_agg.empty:
            row = detail_agg.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Revenue", f"${row['Revenue']:,.0f}")
            c2.metric("Profit", f"${row['Profit']:,.0f}")
            c3.metric("Revenue/Hr", f"${row['Revenue_per_Hour']:,.2f}")
            c4.metric("Margin", f"{row['Profit_Margin'] * 100:.1f}%")
            c5.metric("Hours", f"{row['Billable_Hours']:,.1f}")

        if "YearMonth" in detail_df.columns:
            monthly = detail_df.groupby("YearMonth").agg(
                {"Revenue": "sum", "Cost": "sum", "Billed Hours": "sum"}).reset_index()
            monthly["Profit"] = monthly["Revenue"] - monthly["Cost"]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=monthly["YearMonth"], y=monthly["Revenue"], name="Revenue",
                                           line=dict(color='#636EFA', width=2)))
            fig_trend.add_trace(go.Scatter(x=monthly["YearMonth"], y=monthly["Profit"], name="Profit",
                                           line=dict(color='#00CC96', width=2)))
            fig_trend.update_layout(title=f"Monthly Trend for {selected_group}", xaxis_title="Month",
                                    yaxis_title="Amount ($)",
                                    template="plotly_white", height=300)
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("##### 📊 Individual Entries")
        cols_to_show = ["Worked Date", "Client Name", "Project Name", "Project Category", "Task or Ticket Title",
                        "Resource Name", "Billed Hours", "Hourly Billing Rate", "Revenue", "Cost", "Profit",
                        "Status (Billable / Non-Billable)"]
        cols_to_show = [c for c in cols_to_show if c in detail_df.columns]

        display_df = detail_df[cols_to_show].copy()
        if "Worked Date" in display_df.columns:
            display_df = display_df.sort_values("Worked Date", ascending=False)

        st.dataframe(
            display_df.style.format({
                "Revenue": "${:,.2f}", "Cost": "${:,.2f}", "Profit": "${:,.2f}",
                "Hourly Billing Rate": "${:,.2f}", "Billed Hours": "{:.2f}",
                "Profit_Margin": "{:.1%}" if "Profit_Margin" in display_df.columns else None
            }),
            use_container_width=True, height=400
        )

    st.markdown("---")
    st.markdown("### 💾 Export Data")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Aggregated Report", agg.to_csv(index=False).encode("utf-8"),
                           file_name=f"{group_level.replace(' ', '_').lower()}_profitability.csv", mime="text/csv")
    with col2:
        st.download_button(f"📥 Download {selected_group} Details", detail_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"{selected_group.replace(' ', '_')}_details.csv", mime="text/csv")


# ============================================================================
# PAGE 2: CONSULTANTS VIEW
# ============================================================================
elif view_mode == "Consultants":
    st.title("👥 Consultant Performance Analytics")
    st.markdown("*Analyze consultant productivity, profitability, and utilization*")

    show_global_kpis(df_filtered)
    st.markdown("---")

    group_options = ["Resource Name"]
    if "Consultant Code" in df_filtered.columns:
        group_options.append("Consultant Code")

    group_col = st.selectbox("Group By", group_options, index=0)
    agg_cons = aggregate_profitability(df_filtered, group_col, min_hours)

    # Calculate utilization
    if "Total Hours" in df_filtered.columns and "Billed Hours" in df_filtered.columns:
        hours_by_consultant = df_filtered.groupby(group_col).agg(
            {"Total Hours": "sum", "Billed Hours": "sum"}).reset_index()
        agg_cons = agg_cons.merge(hours_by_consultant[group_col], on=group_col, how="left")

        if "Total Hours" in hours_by_consultant.columns:
            agg_cons = agg_cons.merge(hours_by_consultant[[group_col, "Total Hours"]], on=group_col, how="left",
                                      suffixes=('', '_total'))
            agg_cons["Utilization"] = np.where(agg_cons["Total Hours"] > 0,
                                               (agg_cons["Billable_Hours"] / agg_cons["Total Hours"]) * 100, 0)
    else:
        agg_cons["Utilization"] = np.nan

    if agg_cons.empty:
        st.warning("⚠️ No consultant data after filters.")
        st.stop()

    # Main scatter plot
    fig_scatter_cons = px.scatter(
        agg_cons, x="Revenue_per_Hour", y="Profit_per_Hour", size="Billable_Hours", hover_name=group_col,
        hover_data={"Revenue": ":,.0f", "Billable_Hours": ":,.1f", "Profit": ":,.0f", "Profit_Margin": ":.1%",
                    "Utilization": ":.1f", "Avg_Billing_Rate": ":.2f"},
        color="Utilization", color_continuous_scale="Viridis",
        labels={"Revenue_per_Hour": "Revenue per Hour ($/hr)", "Profit_per_Hour": "Profit per Hour ($/hr)",
                "Utilization": "Utilization (%)"},
        title="Consultant Performance: Revenue vs Profit per Hour", template="plotly_white"
    )
    fig_scatter_cons.update_layout(height=600, margin=dict(l=30, r=30, t=60, b=30))
    st.plotly_chart(fig_scatter_cons, use_container_width=True)

    # Not Billable Revenue Detail
    if status_mode == "Not Billable only":
        st.markdown("---")
        st.markdown("### 🧾 Not Billable – Revenue by Billing Code")

        nb_df = df_filtered.copy()
        if nb_df.empty:
            st.info("No Not Billable records for the current filters.")
        else:
            if "Billing Code Name" in nb_df.columns:
                nb_agg = nb_df.groupby("Billing Code Name", dropna=False).agg(Revenue=("Revenue", "sum")).reset_index()
                total_nb_rev = nb_agg["Revenue"].sum()

                st.metric("Total Not Billable Revenue", f"${total_nb_rev:,.0f}",
                          help="Sum of Revenue for all Not Billable entries under current filters")

                fig_nb = px.bar(nb_agg.sort_values("Revenue", ascending=True), y="Billing Code Name", x="Revenue",
                                orientation="h",
                                title="Revenue by Billing Code (Not Billable)",
                                labels={"Revenue": "Revenue ($)", "Billing Code Name": "Billing Code"},
                                template="plotly_white")
                fig_nb.update_layout(height=500, margin=dict(l=40, r=20, t=60, b=40))
                st.plotly_chart(fig_nb, use_container_width=True)
            else:
                st.warning("Column 'Billing Code Name' not found in the dataset.")

    st.markdown("---")
    st.markdown("### 🏆 Top Consultants")

    top_n_cons = st.slider("Number of consultants to display", 5, 50, 10)
    col1, col2, col3 = st.columns(3)

    # Top by Revenue/Hour
    top_rev_cons = agg_cons.sort_values("Revenue_per_Hour", ascending=False).head(top_n_cons)
    fig_cons_rev = px.bar(top_rev_cons, x=group_col, y="Revenue_per_Hour", title="By Revenue per Hour",
                          labels={"Revenue_per_Hour": "$/hr"}, template="plotly_white",
                          color="Revenue_per_Hour", color_continuous_scale="Blues")
    fig_cons_rev.update_layout(xaxis_tickangle=-45, showlegend=False)
    col1.plotly_chart(fig_cons_rev, use_container_width=True)

    # Top by Profit/Hour
    top_prof_cons = agg_cons.sort_values("Profit_per_Hour", ascending=False).head(top_n_cons)
    fig_cons_prof = px.bar(top_prof_cons, x=group_col, y="Profit_per_Hour", title="By Profit per Hour",
                           labels={"Profit_per_Hour": "$/hr"}, template="plotly_white",
                           color="Profit_per_Hour", color_continuous_scale="Greens")
    fig_cons_prof.update_layout(xaxis_tickangle=-45, showlegend=False)
    col2.plotly_chart(fig_cons_prof, use_container_width=True)

    # Top by Utilization
    top_util_cons = agg_cons.sort_values("Utilization", ascending=False).head(top_n_cons)
    fig_cons_util = px.bar(top_util_cons, x=group_col, y="Utilization", title="By Utilization",
                           labels={"Utilization": "%"}, template="plotly_white",
                           color="Utilization", color_continuous_scale="Oranges")
    fig_cons_util.update_layout(xaxis_tickangle=-45, showlegend=False)
    col3.plotly_chart(fig_cons_util, use_container_width=True)

    # Performance distribution
    st.markdown("---")
    st.markdown("### 📈 Performance Distribution")

    col1, col2 = st.columns(2)
    with col1:
        fig_hist_rate = px.histogram(agg_cons, x="Revenue_per_Hour", nbins=20, title="Revenue per Hour Distribution",
                                     labels={"Revenue_per_Hour": "Revenue/Hour ($)"}, template="plotly_white")
        st.plotly_chart(fig_hist_rate, use_container_width=True)

    with col2:
        fig_hist_util = px.histogram(agg_cons, x="Utilization", nbins=20, title="Utilization Distribution",
                                     labels={"Utilization": "Utilization (%)"}, template="plotly_white")
        st.plotly_chart(fig_hist_util, use_container_width=True)

    # Detailed Performance Table
    st.markdown("---")
    st.markdown("### 📊 Detailed Performance Table")

    display_cols = [group_col, "Revenue", "Cost", "Profit", "Profit_Margin", "Billable_Hours",
                    "Revenue_per_Hour", "Profit_per_Hour", "Utilization", "Avg_Billing_Rate", "ROI"]
    display_cols = [c for c in display_cols if c in agg_cons.columns]

    styled_df = agg_cons[display_cols].sort_values("Profit_per_Hour", ascending=False)

    st.dataframe(
        styled_df.style.format({
            "Revenue": "${:,.0f}", "Cost": "${:,.0f}", "Profit": "${:,.0f}", "Profit_Margin": "{:.1%}",
            "Billable_Hours": "{:.1f}", "Revenue_per_Hour": "${:.2f}", "Profit_per_Hour": "${:.2f}",
            "Utilization": "{:.1f}%", "Avg_Billing_Rate": "${:.2f}", "ROI": "{:.1f}%"
        }).background_gradient(subset=["Profit_Margin"], cmap="RdYlGn"),
        use_container_width=True, height=400
    )

    st.markdown("---")
    st.markdown("### 💾 Export Data")
    st.download_button("📥 Download Consultant Performance Report", agg_cons.to_csv(index=False).encode("utf-8"),
                       file_name=f"consultant_performance_{group_col.replace(' ', '_').lower()}.csv", mime="text/csv")


# ============================================================================
# PAGE 3: TIME SERIES VIEW
# ============================================================================
elif view_mode == "Time Series":
    st.title("📈 Time Series Analysis")
    st.markdown("*Track revenue, profit, and hours over time*")

    show_global_kpis(df_filtered)
    st.markdown("---")

    if "Worked Date" not in df_filtered.columns:
        st.warning("No date information available.")
        st.stop()

    df_ts = df_filtered.dropna(subset=["Worked Date"]).copy()

    col1, col2 = st.columns([1, 2])
    with col1:
        time_grain = st.selectbox("Time Granularity", ["Monthly", "Quarterly", "Weekly"], index=0)

    with col2:
        breakdown_options = ["Overall"]
        if "Client Name" in df_ts.columns:
            breakdown_options.append("Client Name")
        if "Project Name" in df_ts.columns:
            breakdown_options.append("Project Name")
        if "Resource Name" in df_ts.columns:
            breakdown_options.append("Resource Name")
        if "Project Category" in df_ts.columns:
            breakdown_options.append("Project Category")

        breakdown = st.selectbox("Break Down By", breakdown_options, index=0)

    if time_grain == "Monthly":
        time_col = "YearMonth"
    elif time_grain == "Quarterly":
        time_col = "YearQuarter"
    else:
        df_ts["YearWeek"] = df_ts["Worked Date"].dt.to_period("W").dt.to_timestamp()
        time_col = "YearWeek"

    group_cols = [time_col]
    if breakdown != "Overall":
        group_cols.append(breakdown)

    ts_agg = df_ts.groupby(group_cols, as_index=False).agg(Revenue=("Revenue", "sum"), Cost=("Cost", "sum"),
                                                           Billable_Hours=("Billed Hours", "sum"))
    ts_agg["Profit"] = ts_agg["Revenue"] - ts_agg["Cost"]
    ts_agg["Profit_Margin"] = np.where(ts_agg["Revenue"] > 0, ts_agg["Profit"] / ts_agg["Revenue"], np.nan)
    ts_agg["Revenue_per_Hour"] = np.where(ts_agg["Billable_Hours"] > 0, ts_agg["Revenue"] / ts_agg["Billable_Hours"],
                                          np.nan)

    if breakdown == "Overall":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_agg[time_col], y=ts_agg["Revenue"], name="Revenue",
                                 line=dict(color='#636EFA', width=3), mode='lines+markers'))
        fig.add_trace(go.Scatter(x=ts_agg[time_col], y=ts_agg["Profit"], name="Profit",
                                 line=dict(color='#00CC96', width=3), mode='lines+markers'))
        fig.add_trace(go.Scatter(x=ts_agg[time_col], y=ts_agg["Cost"], name="Cost",
                                 line=dict(color='#EF553B', width=3, dash='dash'), mode='lines+markers'))
        fig.update_layout(title=f"{time_grain} Revenue, Profit & Cost Trend", xaxis_title="Period",
                          yaxis_title="Amount ($)",
                          template="plotly_white", height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_hours = px.bar(ts_agg, x=time_col, y="Billable_Hours", title=f"{time_grain} Billed Hours",
                               labels={"Billable_Hours": "Hours"}, template="plotly_white")
            st.plotly_chart(fig_hours, use_container_width=True)

        with col2:
            fig_margin = px.line(ts_agg, x=time_col, y="Profit_Margin", title=f"{time_grain} Profit Margin",
                                 labels={"Profit_Margin": "Margin"}, template="plotly_white")
            fig_margin.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_margin, use_container_width=True)

        if len(ts_agg) > 1:
            st.markdown("---")
            st.markdown("### 📊 Growth Metrics")

            first_period = ts_agg.iloc[0]
            last_period = ts_agg.iloc[-1]

            revenue_growth = ((last_period["Revenue"] - first_period["Revenue"]) / first_period["Revenue"] * 100) if \
            first_period["Revenue"] > 0 else 0
            profit_growth = ((last_period["Profit"] - first_period["Profit"]) / first_period["Profit"] * 100) if \
            first_period["Profit"] > 0 else 0
            hours_growth = ((last_period["Billable_Hours"] - first_period["Billable_Hours"]) / first_period[
                "Billable_Hours"] * 100) if first_period["Billable_Hours"] > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Revenue Growth", f"{revenue_growth:.1f}%")
            col2.metric("Profit Growth", f"{profit_growth:.1f}%")
            col3.metric("Hours Growth", f"{hours_growth:.1f}%")
            col4.metric("Periods", len(ts_agg))

    else:
        entities = sorted(ts_agg[breakdown].dropna().unique().tolist())
        selected_entities = st.multiselect(f"Select {breakdown}(s) to compare", options=entities,
                                           default=entities[:min(5, len(entities))])

        if selected_entities:
            ts_sel = ts_agg[ts_agg[breakdown].isin(selected_entities)].copy()

            fig_rev = px.line(ts_sel, x=time_col, y="Revenue", color=breakdown,
                              title=f"{time_grain} Revenue Comparison",
                              labels={"Revenue": "Revenue ($)"}, template="plotly_white")
            st.plotly_chart(fig_rev, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig_profit = px.line(ts_sel, x=time_col, y="Profit", color=breakdown,
                                     title=f"{time_grain} Profit Comparison",
                                     labels={"Profit": "Profit ($)"}, template="plotly_white")
                st.plotly_chart(fig_profit, use_container_width=True)

            with col2:
                fig_hours = px.line(ts_sel, x=time_col, y="Billable_Hours", color=breakdown,
                                    title=f"{time_grain} Hours Comparison",
                                    labels={"Billable_Hours": "Hours"}, template="plotly_white")
                st.plotly_chart(fig_hours, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💾 Export Time Series Data")
    st.download_button("📥 Download Time Series Report", ts_agg.to_csv(index=False).encode("utf-8"),
                       file_name=f"time_series_{time_grain.lower()}_{breakdown.replace(' ', '_').lower()}.csv",
                       mime="text/csv")


# ============================================================================
# PAGE 4: EXECUTIVE SUMMARY
# ============================================================================
elif view_mode == "Executive Summary":
    st.title("🎯 Executive Summary Dashboard")
    st.markdown("*High-level overview of business performance*")

    show_global_kpis(df_filtered)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👥 Workforce")
        st.metric("Active Consultants",
                  df_filtered["Resource Name"].nunique() if "Resource Name" in df_filtered.columns else 0)
        if "Total Hours" in df_filtered.columns:
            total_hours = df_filtered["Total Hours"].sum()
            billable_hours = df_filtered["Billed Hours"].sum()
            util = (billable_hours / total_hours * 100) if total_hours > 0 else 0
            st.metric("Overall Utilization", f"{util:.1f}%")

    with col2:
        st.markdown("### 🤝 Clients")
        st.metric("Active Clients", df_filtered["Client Name"].nunique() if "Client Name" in df_filtered.columns else 0)
        st.metric("Active Projects",
                  df_filtered["Project Name"].nunique() if "Project Name" in df_filtered.columns else 0)

    with col3:
        st.markdown("### 💰 Financial")
        avg_rate = df_filtered["Hourly Billing Rate"].mean() if "Hourly Billing Rate" in df_filtered.columns else 0
        st.metric("Avg Billing Rate", f"${avg_rate:.2f}/hr")
        if "Realization_Rate" in df_filtered.columns:
            real_rate = df_filtered["Realization_Rate"].mean() * 100
            st.metric("Realization Rate", f"{real_rate:.1f}%")

    st.markdown("---")
    st.markdown("### 🏆 Top Performers")

    tab1, tab2, tab3 = st.tabs(["By Client", "By Project", "By Consultant"])

    with tab1:
        if "Client Name" in df_filtered.columns:
            client_agg = aggregate_profitability(df_filtered, "Client Name", 0).head(10)
            if not client_agg.empty:
                fig = px.bar(client_agg, y="Client Name", x="Revenue", orientation='h', color="Profit_Margin",
                             title="Top 10 Clients by Revenue", template="plotly_white",
                             color_continuous_scale="RdYlGn")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No client data available")

    with tab2:
        if "Project Name" in df_filtered.columns:
            project_agg = aggregate_profitability(df_filtered, "Project Name", 0).head(10)
            if not project_agg.empty:
                fig = px.bar(project_agg, y="Project Name", x="Revenue", orientation='h', color="Profit_Margin",
                             title="Top 10 Projects by Revenue", template="plotly_white",
                             color_continuous_scale="RdYlGn")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No project data available")

    with tab3:
        if "Resource Name" in df_filtered.columns:
            consultant_agg = aggregate_profitability(df_filtered, "Resource Name", 0).head(10)
            if not consultant_agg.empty:
                fig = px.bar(consultant_agg, y="Resource Name", x="Revenue", orientation='h', color="Profit_Margin",
                             title="Top 10 Consultants by Revenue", template="plotly_white",
                             color_continuous_scale="RdYlGn")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No consultant data available")

    st.markdown("---")
    st.markdown("### 📈 Recent Trends")

    if "YearMonth" in df_filtered.columns:
        monthly = df_filtered.groupby("YearMonth").agg(
            {"Revenue": "sum", "Cost": "sum", "Billed Hours": "sum"}).reset_index().tail(12)
        monthly["Profit"] = monthly["Revenue"] - monthly["Cost"]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly["YearMonth"], y=monthly["Revenue"], name="Revenue", marker_color='#636EFA'))
        fig.add_trace(go.Bar(x=monthly["YearMonth"], y=monthly["Cost"], name="Cost", marker_color='#EF553B'))
        fig.add_trace(go.Scatter(x=monthly["YearMonth"], y=monthly["Profit"], name="Profit",
                                 mode='lines+markers', line=dict(color='#00CC96', width=3)))
        fig.update_layout(title="Last 12 Months Performance", xaxis_title="Month", yaxis_title="Amount ($)",
                          template="plotly_white", height=400, barmode='group')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Steeves & Associates Revenue & Profitability Dashboard v2.0</p>
        <p style='font-size: 0.8em;'>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    """,
    unsafe_allow_html=True
)
