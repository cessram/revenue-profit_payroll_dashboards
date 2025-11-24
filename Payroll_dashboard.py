import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Inland – Salary & Payroll Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with red theme
st.markdown("""
    <style>
    .metric-card {
        background-color: #fff5f5;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(220, 38, 38, 0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc2626;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #fef2f2;
        border-radius: 4px 4px 0 0;
        color: #991b1b;
    }
    .stTabs [aria-selected="true"] {
        background-color: #dc2626;
        color: white;
    }
    h1, h2, h3 {
        color: #991b1b;
    }
    .stDownloadButton button {
        background-color: #dc2626;
        color: white;
    }
    .stDownloadButton button:hover {
        background-color: #b91c1c;
    }
    </style>
    """, unsafe_allow_html=True)

DEFAULT_DATA_PATH = "Inland Payroll Database.xlsx"


# -------------------------
# DATA LOADING + CLEANING
# -------------------------
@st.cache_data
def load_data(file_source) -> pd.DataFrame:
    """Load data from file path or uploaded file"""
    try:
        if isinstance(file_source, str):
            df = pd.read_excel(file_source)
        else:
            df = pd.read_excel(file_source)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Define numeric columns
    numeric_cols = [
        "Annual Salary", "Remumenration", "Pay Per Period",
        "Hourly Rate", "Std Hours", "Monthly salary",
        "Monthly commission", "Vehicle Allowance",
        "Estimated Budget", "Percentage Conversion",
    ]

    # Convert to numeric
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate Pay Per Period
    if {"Hourly Rate", "Std Hours"}.issubset(df.columns):
        df["Pay Per Period"] = df["Hourly Rate"] * df["Std Hours"]

    # Calculate Monthly salary from Annual
    if "Annual Salary" in df.columns:
        if "Monthly salary" in df.columns:
            missing_monthly = df["Monthly salary"].isna() | (df["Monthly salary"] == 0)
            df.loc[missing_monthly, "Monthly salary"] = df["Annual Salary"] / 12.0
        else:
            df["Monthly salary"] = df["Annual Salary"] / 12.0

    # Normalize Percentage Conversion
    if "Percentage Conversion" in df.columns:
        df["Percentage Conversion"] = pd.to_numeric(
            df["Percentage Conversion"], errors="coerce"
        )
        mask_pct = df["Percentage Conversion"] > 1
        df.loc[mask_pct, "Percentage Conversion"] = (
                df.loc[mask_pct, "Percentage Conversion"] / 100.0
        )

    # Calculate Total Compensation
    comp_cols = ["Monthly salary", "Monthly commission", "Vehicle Allowance"]
    available_comp_cols = [c for c in comp_cols if c in df.columns]
    if available_comp_cols:
        df["Total Monthly Compensation"] = df[available_comp_cols].sum(axis=1)
        df["Total Annual Compensation"] = df["Total Monthly Compensation"] * 12

    # Calculate Budget Utilization
    if "Annual Salary" in df.columns and "Estimated Budget" in df.columns:
        df["Budget Utilization %"] = (df["Annual Salary"] / df["Estimated Budget"] * 100).round(2)

    return df


# -------------------------
# SIDEBAR – FILE UPLOAD & FILTERS
# -------------------------
st.sidebar.title("📁 Data Source")

# Download Template Button
st.sidebar.markdown("### 📋 Template File")
st.sidebar.markdown("Download the Excel template to ensure your data follows the correct format.")


# Create template Excel file
def create_template():
    template_data = {
        'Employee Name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'Department': ['Engineering', 'Sales', 'HR'],
        'Position': ['Software Engineer', 'Sales Manager', 'HR Specialist'],
        'Hiring Category': ['Full-Time', 'Full-Time', 'Part-Time'],
        'Annual Salary': [85000, 75000, 55000],
        'Remumenration': [85000, 75000, 55000],
        'Pay Per Period': [3269.23, 2884.62, 2115.38],
        'Hourly Rate': [40.87, 36.06, 26.44],
        'Std Hours': [80, 80, 80],
        'Monthly salary': [7083.33, 6250.00, 4583.33],
        'Monthly commission': [0, 1500, 0],
        'Vehicle Allowance': [500, 400, 0],
        'Estimated Budget': [90000, 80000, 60000],
        'Percentage Conversion': [0.80, 0.85, 0.75]
    }

    template_df = pd.DataFrame(template_data)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Payroll Template')

        # Auto-adjust columns width
        worksheet = writer.sheets['Payroll Template']
        for idx, col in enumerate(template_df.columns):
            max_length = max(
                template_df[col].astype(str).apply(len).max(),
                len(col)
            )
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)

    return output.getvalue()


template_excel = create_template()

st.sidebar.download_button(
    label="⬇️ Download Excel Template",
    data=template_excel,
    file_name="Inland_Payroll_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Download this template and fill in your employee data"
)

# Template instructions
with st.sidebar.expander("📖 Template Column Guide"):
    st.markdown("""
    **Required Columns:**
    - `Employee Name`: Full name of employee
    - `Department`: Department name
    - `Position`: Job title/position
    - `Hiring Category`: Full-Time, Part-Time, Contract, etc.
    - `Annual Salary`: Yearly salary amount
    - `Remumenration`: Total compensation
    - `Hourly Rate`: Hourly pay rate
    - `Std Hours`: Standard hours per period
    - `Monthly salary`: Monthly base salary
    - `Estimated Budget`: Budget allocated

    **Optional Columns:**
    - `Pay Per Period`: Auto-calculated (Hourly Rate × Std Hours)
    - `Monthly commission`: Commission amount
    - `Vehicle Allowance`: Vehicle allowance
    - `Percentage Conversion`: Conversion rate (0-1 or 0-100)

    **Tips:**
    - Delete sample rows and add your data
    - Keep column headers exactly as shown
    - Use numbers without currency symbols
    - Percentage: Use 0.80 for 80% or 80
    """)

st.sidebar.markdown("---")

# File upload option
upload_option = st.sidebar.radio(
    "Choose data source:",
    ["Use Default File", "Upload Custom File"]
)

if upload_option == "Upload Custom File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"],
        help="Upload your payroll data file"
    )
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if not df.empty:
            st.sidebar.success("✅ Custom file loaded successfully!")

            # Validate columns
            expected_cols = [
                'Employee Name', 'Department', 'Position', 'Hiring Category',
                'Annual Salary', 'Remumenration', 'Pay Per Period', 'Hourly Rate',
                'Std Hours', 'Monthly salary', 'Monthly commission',
                'Vehicle Allowance', 'Estimated Budget', 'Percentage Conversion'
            ]

            missing_cols = [col for col in expected_cols if col not in df.columns]
            extra_cols = [col for col in df.columns if col not in expected_cols]

            if missing_cols or extra_cols:
                with st.sidebar.expander("⚠️ Column Validation", expanded=False):
                    if missing_cols:
                        st.warning(f"**Missing columns:** {', '.join(missing_cols[:3])}" +
                                   (f" and {len(missing_cols) - 3} more" if len(missing_cols) > 3 else ""))
                    if extra_cols:
                        st.info(f"**Extra columns found:** {', '.join(extra_cols[:3])}" +
                                (f" and {len(extra_cols) - 3} more" if len(extra_cols) > 3 else ""))
                    st.caption("The dashboard will work with available columns.")
            else:
                st.sidebar.success("✅ All columns validated!")
        else:
            st.sidebar.error("❌ Error loading file. Please check the format.")
    else:
        st.sidebar.info("Please upload a file to proceed")
        df = pd.DataFrame()
else:
    try:
        df = load_data(DEFAULT_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Default file '{DEFAULT_DATA_PATH}' not found. Please upload a custom file.")
        df = pd.DataFrame()

# Proceed only if data is loaded
if not df.empty:
    st.sidebar.markdown("---")
    st.sidebar.title("🔍 Filters")

    # Department filter
    if "Department" in df.columns:
        dept_list = sorted(df["Department"].dropna().unique())
        selected_depts = st.sidebar.multiselect(
            "Department",
            options=dept_list,
            default=dept_list,
        )
    else:
        selected_depts = []

    # Position filter
    if "Position" in df.columns:
        pos_list = sorted(df["Position"].dropna().unique())
        selected_positions = st.sidebar.multiselect(
            "Position",
            options=pos_list,
            default=pos_list,
        )
    else:
        selected_positions = []

    # Hiring Category filter
    if "Hiring Category" in df.columns:
        cat_list = sorted(df["Hiring Category"].dropna().unique())
        selected_hiring_cat = st.sidebar.multiselect(
            "Hiring Category",
            options=cat_list,
            default=cat_list,
        )
    else:
        selected_hiring_cat = []

    # Annual Salary range filter
    salary_range = None
    if "Annual Salary" in df.columns:
        salary_series = df["Annual Salary"].dropna()
        if not salary_series.empty:
            min_salary = float(salary_series.min())
            max_salary = float(salary_series.max())
            salary_range = st.sidebar.slider(
                "Annual Salary Range",
                min_value=int(min_salary),
                max_value=int(max_salary),
                value=(int(min_salary), int(max_salary)),
                step=1000,
            )

    # Employee search
    employee_search = st.sidebar.text_input("🔎 Search Employee Name")

    # Reset filters button
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()

    # -------------------------
    # APPLY FILTERS
    # -------------------------
    filtered_df = df.copy()

    if selected_depts and "Department" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Department"].isin(selected_depts)]

    if selected_positions and "Position" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Position"].isin(selected_positions)]

    if selected_hiring_cat and "Hiring Category" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Hiring Category"].isin(selected_hiring_cat)]

    if salary_range and "Annual Salary" in filtered_df.columns:
        low, high = salary_range
        filtered_df = filtered_df[filtered_df["Annual Salary"].between(low, high)]

    if employee_search and "Employee Name" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Employee Name"]
            .fillna("")
            .str.contains(employee_search, case=False, na=False)
        ]

    # -------------------------
    # MAIN LAYOUT
    # -------------------------
    st.title("📊 Inland – Salary, Compensation & Payroll Dashboard")
    st.caption(
        f"Interactive analytics for {len(filtered_df)} employees | Last updated: {datetime.now().strftime('%B %d, %Y')}"
    )

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview",
        "👥 Department Analysis",
        "💰 Compensation Details",
        "🏢 By Department",
        "👤 By Employee",
        "📋 Data Explorer"
    ])

    # -------------------------
    # TAB 1: OVERVIEW
    # -------------------------
    with tab1:
        st.markdown("### Key Payroll Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        # Headcount
        headcount = filtered_df["Employee Name"].nunique() if "Employee Name" in filtered_df.columns else 0
        with col1:
            st.metric("👥 Headcount", f"{headcount:,}")

        # Total Annual Payroll
        total_annual = filtered_df["Annual Salary"].sum() if "Annual Salary" in filtered_df.columns else 0
        with col2:
            st.metric("💵 Total Annual Payroll", f"${total_annual:,.0f}")

        # Average Annual Salary
        avg_annual = filtered_df["Annual Salary"].mean() if "Annual Salary" in filtered_df.columns else 0
        with col3:
            st.metric("📊 Avg Annual Salary", f"${avg_annual:,.0f}")

        # Median Annual Salary
        median_annual = filtered_df["Annual Salary"].median() if "Annual Salary" in filtered_df.columns else 0
        with col4:
            st.metric("📈 Median Salary", f"${median_annual:,.0f}")

        # Total Budget
        total_budget = filtered_df["Estimated Budget"].sum() if "Estimated Budget" in filtered_df.columns else 0
        with col5:
            st.metric("🎯 Total Budget", f"${total_budget:,.0f}")

        # Budget utilization
        if total_budget > 0:
            budget_util = (total_annual / total_budget * 100)
            st.markdown(f"**Budget Utilization:** {budget_util:.1f}%")

            # Create custom progress bar with red color
            progress_html = f"""
            <div style="width: 100%; background-color: #fee2e2; border-radius: 10px; height: 25px;">
                <div style="width: {min(budget_util, 100)}%; background-color: #dc2626; 
                     height: 25px; border-radius: 10px; text-align: center; line-height: 25px; 
                     color: white; font-weight: bold;">
                    {budget_util:.1f}%
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)

        st.markdown("---")

        # Visualization row 1
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if "Department" in filtered_df.columns and "Annual Salary" in filtered_df.columns:
                dept_salary = (
                    filtered_df.groupby("Department", as_index=False)["Annual Salary"]
                    .mean()
                    .sort_values("Annual Salary", ascending=False)
                )
                fig_dept = px.bar(
                    dept_salary,
                    x="Department",
                    y="Annual Salary",
                    title="Average Annual Salary by Department",
                    color="Annual Salary",
                    color_continuous_scale="Reds",
                )
                fig_dept.update_layout(
                    xaxis_title="",
                    yaxis_tickprefix="$",
                    showlegend=False
                )
                st.plotly_chart(fig_dept, use_container_width=True)

        with chart_col2:
            if "Hiring Category" in filtered_df.columns:
                cat_counts = filtered_df["Hiring Category"].value_counts().reset_index()
                cat_counts.columns = ["Hiring Category", "Count"]
                fig_cat = px.pie(
                    cat_counts,
                    values="Count",
                    names="Hiring Category",
                    title="Employee Distribution by Hiring Category",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")

        # Visualization row 2
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            if "Annual Salary" in filtered_df.columns:
                fig_hist = px.histogram(
                    filtered_df,
                    x="Annual Salary",
                    nbins=30,
                    title="Salary Distribution",
                    labels={"Annual Salary": "Annual Salary"},
                    color_discrete_sequence=["#dc2626"]
                )
                fig_hist.update_layout(
                    xaxis_tickprefix="$",
                    yaxis_title="Number of Employees"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

        with chart_col4:
            if "Position" in filtered_df.columns:
                top_positions = (
                    filtered_df.groupby("Position", as_index=False)
                    .size()
                    .sort_values("size", ascending=False)
                    .head(10)
                )
                top_positions.columns = ["Position", "Count"]
                fig_pos_count = px.bar(
                    top_positions,
                    x="Count",
                    y="Position",
                    orientation="h",
                    title="Top 10 Positions by Headcount",
                    color="Count",
                    color_continuous_scale="Reds"
                )
                fig_pos_count.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_pos_count, use_container_width=True)

    # -------------------------
    # TAB 2: DEPARTMENT ANALYSIS
    # -------------------------
    with tab2:
        st.markdown("### Department-Level Analysis")

        if "Department" in filtered_df.columns:
            # Department summary table
            dept_summary = filtered_df.groupby("Department").agg({
                "Employee Name": "nunique",
                "Annual Salary": ["sum", "mean", "median"],
                "Estimated Budget": "sum"
            }).round(2)

            dept_summary.columns = ["Headcount", "Total Salary", "Avg Salary", "Median Salary", "Total Budget"]
            dept_summary["Budget Util %"] = (dept_summary["Total Salary"] / dept_summary["Total Budget"] * 100).round(1)
            dept_summary = dept_summary.sort_values("Total Salary", ascending=False)

            st.markdown("#### Department Summary Metrics")
            st.dataframe(
                dept_summary.style.format({
                    "Headcount": "{:,.0f}",
                    "Total Salary": "${:,.0f}",
                    "Avg Salary": "${:,.0f}",
                    "Median Salary": "${:,.0f}",
                    "Total Budget": "${:,.0f}",
                    "Budget Util %": "{:.1f}%"
                }),
                use_container_width=True
            )

            st.markdown("---")

            # Visualizations
            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                # Compensation breakdown by department
                comp_cols = [c for c in ["Monthly salary", "Monthly commission", "Vehicle Allowance"]
                             if c in filtered_df.columns]

                if comp_cols:
                    comp_summary = (
                        filtered_df.groupby("Department")[comp_cols]
                        .sum()
                        .reset_index()
                        .melt(id_vars="Department", value_vars=comp_cols,
                              var_name="Component", value_name="Amount")
                    )

                    fig_comp = px.bar(
                        comp_summary,
                        x="Department",
                        y="Amount",
                        color="Component",
                        barmode="stack",
                        title="Monthly Compensation Breakdown by Department",
                        color_discrete_sequence=px.colors.sequential.Reds_r
                    )
                    fig_comp.update_layout(xaxis_title="", yaxis_tickprefix="$")
                    st.plotly_chart(fig_comp, use_container_width=True)

            with viz_col2:
                # Headcount by department
                dept_headcount = (
                    filtered_df.groupby("Department")["Employee Name"]
                    .nunique()
                    .reset_index()
                    .sort_values("Employee Name", ascending=False)
                )
                dept_headcount.columns = ["Department", "Headcount"]

                fig_dept_hc = px.bar(
                    dept_headcount,
                    x="Department",
                    y="Headcount",
                    title="Headcount by Department",
                    color="Headcount",
                    color_continuous_scale="Reds"
                )
                fig_dept_hc.update_layout(xaxis_title="", showlegend=False)
                st.plotly_chart(fig_dept_hc, use_container_width=True)

            # Salary box plot by department
            if "Annual Salary" in filtered_df.columns:
                st.markdown("#### Salary Distribution Across Departments")
                fig_box = px.box(
                    filtered_df,
                    x="Department",
                    y="Annual Salary",
                    title="Salary Range by Department",
                    points="outliers",
                    color_discrete_sequence=["#dc2626"]
                )
                fig_box.update_layout(yaxis_tickprefix="$")
                st.plotly_chart(fig_box, use_container_width=True)

    # -------------------------
    # TAB 3: COMPENSATION DETAILS
    # -------------------------
    with tab3:
        st.markdown("### Detailed Compensation Analysis")

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            if "Position" in filtered_df.columns and "Annual Salary" in filtered_df.columns:
                pos_salary = (
                    filtered_df.groupby("Position", as_index=False)["Annual Salary"]
                    .sum()
                    .sort_values("Annual Salary", ascending=False)
                    .head(15)
                )
                fig_pos = px.bar(
                    pos_salary,
                    y="Position",
                    x="Annual Salary",
                    orientation="h",
                    title="Total Annual Salary by Position (Top 15)",
                    color="Annual Salary",
                    color_continuous_scale="Reds"
                )
                fig_pos.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_tickprefix="$",
                    showlegend=False
                )
                st.plotly_chart(fig_pos, use_container_width=True)

        with comp_col2:
            if "Hourly Rate" in filtered_df.columns and "Std Hours" in filtered_df.columns:
                hourly_data = filtered_df[
                    filtered_df["Hourly Rate"].notna() &
                    (filtered_df["Hourly Rate"] > 0)
                    ].copy()

                if not hourly_data.empty:
                    fig_scatter = px.scatter(
                        hourly_data,
                        x="Std Hours",
                        y="Hourly Rate",
                        color="Department" if "Department" in hourly_data.columns else None,
                        size="Annual Salary" if "Annual Salary" in hourly_data.columns else None,
                        hover_data=["Employee Name"] if "Employee Name" in hourly_data.columns else None,
                        title="Hourly Rate vs Standard Hours",
                        color_discrete_sequence=px.colors.sequential.Reds_r
                    )
                    fig_scatter.update_layout(yaxis_tickprefix="$")
                    st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")

        # Compensation summary statistics
        if "Total Annual Compensation" in filtered_df.columns:
            st.markdown("#### Total Compensation Statistics")

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

            total_comp = filtered_df["Total Annual Compensation"].sum()
            avg_comp = filtered_df["Total Annual Compensation"].mean()
            median_comp = filtered_df["Total Annual Compensation"].median()
            p90_comp = filtered_df["Total Annual Compensation"].quantile(0.9)

            with stat_col1:
                st.metric("Total Compensation", f"${total_comp:,.0f}")
            with stat_col2:
                st.metric("Average", f"${avg_comp:,.0f}")
            with stat_col3:
                st.metric("Median", f"${median_comp:,.0f}")
            with stat_col4:
                st.metric("90th Percentile", f"${p90_comp:,.0f}")

        # Top earners
        if "Annual Salary" in filtered_df.columns and "Employee Name" in filtered_df.columns:
            st.markdown("#### Top 10 Earners")
            top_earners = filtered_df.nlargest(10, "Annual Salary")[
                ["Employee Name", "Department", "Position", "Annual Salary"]
            ].reset_index(drop=True)
            top_earners.index += 1
            st.dataframe(
                top_earners.style.format({"Annual Salary": "${:,.0f}"}),
                use_container_width=True
            )

    # -------------------------
    # TAB 4: BY DEPARTMENT
    # -------------------------
    with tab4:
        st.markdown("### Department-Specific Analysis")

        if "Department" in filtered_df.columns and not filtered_df.empty:
            # Get list of departments
            departments = sorted(filtered_df["Department"].dropna().unique())

            if departments:
                # Create sub-tabs for each department
                dept_tabs = st.tabs([f"📁 {dept}" for dept in departments])

                for idx, dept in enumerate(departments):
                    with dept_tabs[idx]:
                        dept_data = filtered_df[filtered_df["Department"] == dept]

                        # Department KPIs
                        st.markdown(f"#### {dept} Department Overview")

                        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

                        dept_headcount = dept_data[
                            "Employee Name"].nunique() if "Employee Name" in dept_data.columns else 0
                        dept_total_salary = dept_data[
                            "Annual Salary"].sum() if "Annual Salary" in dept_data.columns else 0
                        dept_avg_salary = dept_data[
                            "Annual Salary"].mean() if "Annual Salary" in dept_data.columns else 0
                        dept_budget = dept_data[
                            "Estimated Budget"].sum() if "Estimated Budget" in dept_data.columns else 0

                        with kpi_col1:
                            st.metric("Headcount", f"{dept_headcount:,}")
                        with kpi_col2:
                            st.metric("Total Salary", f"${dept_total_salary:,.0f}")
                        with kpi_col3:
                            st.metric("Avg Salary", f"${dept_avg_salary:,.0f}")
                        with kpi_col4:
                            if dept_budget > 0:
                                dept_util = (dept_total_salary / dept_budget * 100)
                                st.metric("Budget Util", f"{dept_util:.1f}%")
                            else:
                                st.metric("Budget", f"${dept_budget:,.0f}")

                        st.markdown("---")

                        # Department visualizations
                        viz_col1, viz_col2 = st.columns(2)

                        with viz_col1:
                            # Position breakdown
                            if "Position" in dept_data.columns:
                                pos_counts = dept_data["Position"].value_counts().head(10).reset_index()
                                pos_counts.columns = ["Position", "Count"]

                                fig_pos = px.bar(
                                    pos_counts,
                                    x="Position",
                                    y="Count",
                                    title=f"Top Positions in {dept}",
                                    color="Count",
                                    color_continuous_scale="Reds"
                                )
                                fig_pos.update_layout(showlegend=False, xaxis_title="")
                                st.plotly_chart(fig_pos, use_container_width=True)

                        with viz_col2:
                            # Salary distribution
                            if "Annual Salary" in dept_data.columns:
                                fig_salary_dist = px.histogram(
                                    dept_data,
                                    x="Annual Salary",
                                    nbins=20,
                                    title=f"Salary Distribution in {dept}",
                                    color_discrete_sequence=["#dc2626"]
                                )
                                fig_salary_dist.update_layout(
                                    xaxis_tickprefix="$",
                                    yaxis_title="Count"
                                )
                                st.plotly_chart(fig_salary_dist, use_container_width=True)

                        # Hiring category breakdown
                        if "Hiring Category" in dept_data.columns:
                            st.markdown(f"##### Hiring Categories in {dept}")
                            cat_breakdown = dept_data["Hiring Category"].value_counts().reset_index()
                            cat_breakdown.columns = ["Hiring Category", "Count"]

                            fig_cat = px.pie(
                                cat_breakdown,
                                values="Count",
                                names="Hiring Category",
                                title=f"Hiring Category Distribution",
                                hole=0.3,
                                color_discrete_sequence=px.colors.sequential.Reds_r
                            )
                            st.plotly_chart(fig_cat, use_container_width=True)

                        st.markdown("---")

                        # Employee list for this department
                        st.markdown(f"##### Employees in {dept}")

                        display_cols = [
                            c for c in ["Employee Name", "Position", "Hiring Category",
                                        "Annual Salary", "Hourly Rate", "Monthly salary"]
                            if c in dept_data.columns
                        ]

                        if display_cols:
                            st.dataframe(
                                dept_data[display_cols].sort_values(
                                    by="Annual Salary" if "Annual Salary" in display_cols else display_cols[0],
                                    ascending=False
                                ).style.format({
                                                   "Annual Salary": "${:,.0f}",
                                                   "Hourly Rate": "${:,.2f}",
                                                   "Monthly salary": "${:,.0f}"
                                               } if "Annual Salary" in display_cols else {}),
                                use_container_width=True,
                                height=400
                            )
            else:
                st.info("No department data available.")
        else:
            st.warning("Department column not found in the dataset.")

    # -------------------------
    # TAB 5: BY EMPLOYEE
    # -------------------------
    with tab5:
        st.markdown("### Individual Employee Analysis")

        if "Employee Name" in filtered_df.columns and not filtered_df.empty:
            # Employee selector
            employees = sorted(filtered_df["Employee Name"].dropna().unique())

            if employees:
                selected_employee = st.selectbox(
                    "🔍 Select an employee to view details:",
                    options=employees,
                    index=0
                )

                if selected_employee:
                    emp_data = filtered_df[filtered_df["Employee Name"] == selected_employee].iloc[0]

                    st.markdown(f"## 👤 {selected_employee}")
                    st.markdown("---")

                    # Employee profile
                    profile_col1, profile_col2, profile_col3 = st.columns(3)

                    with profile_col1:
                        st.markdown("#### 📋 Basic Information")
                        if "Department" in emp_data.index:
                            st.write(f"**Department:** {emp_data['Department']}")
                        if "Position" in emp_data.index:
                            st.write(f"**Position:** {emp_data['Position']}")
                        if "Hiring Category" in emp_data.index:
                            st.write(f"**Hiring Category:** {emp_data['Hiring Category']}")

                    with profile_col2:
                        st.markdown("#### 💰 Compensation")
                        if "Annual Salary" in emp_data.index:
                            st.write(f"**Annual Salary:** ${emp_data['Annual Salary']:,.0f}")
                        if "Monthly salary" in emp_data.index:
                            st.write(f"**Monthly Salary:** ${emp_data['Monthly salary']:,.0f}")
                        if "Hourly Rate" in emp_data.index and pd.notna(emp_data['Hourly Rate']):
                            st.write(f"**Hourly Rate:** ${emp_data['Hourly Rate']:,.2f}")

                    with profile_col3:
                        st.markdown("#### 📊 Additional Details")
                        if "Std Hours" in emp_data.index and pd.notna(emp_data['Std Hours']):
                            st.write(f"**Standard Hours:** {emp_data['Std Hours']:.0f}")
                        if "Pay Per Period" in emp_data.index and pd.notna(emp_data['Pay Per Period']):
                            st.write(f"**Pay Per Period:** ${emp_data['Pay Per Period']:,.2f}")
                        if "Vehicle Allowance" in emp_data.index and pd.notna(emp_data['Vehicle Allowance']):
                            st.write(f"**Vehicle Allowance:** ${emp_data['Vehicle Allowance']:,.0f}")

                    st.markdown("---")

                    # Compensation breakdown visualization
                    comp_components = {}
                    if "Monthly salary" in emp_data.index and pd.notna(emp_data['Monthly salary']):
                        comp_components["Base Salary"] = emp_data['Monthly salary']
                    if "Monthly commission" in emp_data.index and pd.notna(emp_data['Monthly commission']):
                        comp_components["Commission"] = emp_data['Monthly commission']
                    if "Vehicle Allowance" in emp_data.index and pd.notna(emp_data['Vehicle Allowance']):
                        comp_components["Vehicle Allowance"] = emp_data['Vehicle Allowance']

                    if comp_components:
                        st.markdown("#### Monthly Compensation Breakdown")

                        comp_df = pd.DataFrame({
                            "Component": list(comp_components.keys()),
                            "Amount": list(comp_components.values())
                        })

                        fig_emp_comp = px.pie(
                            comp_df,
                            values="Amount",
                            names="Component",
                            title=f"Compensation Components for {selected_employee}",
                            hole=0.4,
                            color_discrete_sequence=px.colors.sequential.Reds_r
                        )
                        st.plotly_chart(fig_emp_comp, use_container_width=True)

                    st.markdown("---")

                    # Comparison with department and company
                    st.markdown("#### 📈 Comparative Analysis")

                    if "Annual Salary" in emp_data.index and "Annual Salary" in filtered_df.columns:
                        comp_col1, comp_col2 = st.columns(2)

                        with comp_col1:
                            # Compare with department
                            if "Department" in emp_data.index:
                                dept_avg = filtered_df[
                                    filtered_df["Department"] == emp_data["Department"]
                                    ]["Annual Salary"].mean()

                                comparison_data = pd.DataFrame({
                                    "Category": [selected_employee, f"{emp_data['Department']} Avg"],
                                    "Salary": [emp_data['Annual Salary'], dept_avg]
                                })

                                fig_dept_comp = px.bar(
                                    comparison_data,
                                    x="Category",
                                    y="Salary",
                                    title=f"Salary vs Department Average",
                                    color="Category",
                                    text="Salary",
                                    color_discrete_sequence=["#dc2626", "#ef4444"]
                                )
                                fig_dept_comp.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                                fig_dept_comp.update_layout(yaxis_tickprefix="$", showlegend=False)
                                st.plotly_chart(fig_dept_comp, use_container_width=True)

                        with comp_col2:
                            # Compare with company
                            company_avg = filtered_df["Annual Salary"].mean()
                            company_median = filtered_df["Annual Salary"].median()

                            comparison_data = pd.DataFrame({
                                "Category": [selected_employee, "Company Avg", "Company Median"],
                                "Salary": [emp_data['Annual Salary'], company_avg, company_median]
                            })

                            fig_company_comp = px.bar(
                                comparison_data,
                                x="Category",
                                y="Salary",
                                title=f"Salary vs Company Benchmarks",
                                color="Category",
                                text="Salary",
                                color_discrete_sequence=["#dc2626", "#ef4444", "#f87171"]
                            )
                            fig_company_comp.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                            fig_company_comp.update_layout(yaxis_tickprefix="$", showlegend=False)
                            st.plotly_chart(fig_company_comp, use_container_width=True)

                    st.markdown("---")

                    # Full employee record
                    st.markdown("#### Complete Employee Record")
                    emp_full_data = pd.DataFrame({
                        "Field": emp_data.index,
                        "Value": emp_data.values
                    })
                    st.dataframe(emp_full_data, use_container_width=True, height=400)
            else:
                st.info("No employee data available.")
        else:
            st.warning("Employee Name column not found in the dataset.")

    # -------------------------
    # TAB 6: DATA EXPLORER
    # -------------------------
    with tab6:
        st.markdown("### Employee-Level Payroll Details")

        # Download buttons
        if not filtered_df.empty:
            dl_col1, dl_col2 = st.columns(2)

            # CSV Download
            with dl_col1:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv,
                    file_name=f"payroll_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

            # Excel Download
            with dl_col2:
                # Create Excel file in memory
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, index=False, sheet_name='Payroll Data')

                    # Auto-adjust columns width
                    worksheet = writer.sheets['Payroll Data']
                    for idx, col in enumerate(filtered_df.columns):
                        max_length = max(
                            filtered_df[col].astype(str).apply(len).max(),
                            len(col)
                        )
                        worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)

                excel_data = output.getvalue()

                st.download_button(
                    label="📊 Download Filtered Data (Excel)",
                    data=excel_data,
                    file_name=f"payroll_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        # Column selector
        if not filtered_df.empty:
            all_columns = filtered_df.columns.tolist()
            default_cols = [
                "Employee Name", "Department", "Position", "Hiring Category",
                "Annual Salary", "Monthly salary", "Hourly Rate", "Std Hours",
                "Pay Per Period", "Estimated Budget"
            ]
            default_display = [c for c in default_cols if c in all_columns]

            selected_cols = st.multiselect(
                "Select columns to display:",
                options=all_columns,
                default=default_display
            )

            if selected_cols:
                st.dataframe(
                    filtered_df[selected_cols].sort_values(
                        by=[c for c in ["Department", "Employee Name"] if c in selected_cols]
                    ),
                    use_container_width=True,
                    height=600
                )
            else:
                st.warning("Please select at least one column to display.")

            # Data quality summary
            st.markdown("---")
            st.markdown("#### Data Quality Summary")

            quality_col1, quality_col2 = st.columns(2)

            with quality_col1:
                st.markdown("**Missing Values:**")
                missing = filtered_df.isnull().sum()
                missing_pct = (missing / len(filtered_df) * 100).round(1)
                missing_df = pd.DataFrame({
                    "Column": missing.index,
                    "Missing Count": missing.values,
                    "Missing %": missing_pct.values
                })
                missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
                    "Missing Count", ascending=False
                )
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True, height=300)
                else:
                    st.success("No missing values detected!")

            with quality_col2:
                st.markdown("**Data Summary:**")
                st.write(f"Total Records: {len(filtered_df):,}")
                st.write(f"Total Columns: {len(filtered_df.columns)}")
                if "Department" in filtered_df.columns:
                    st.write(f"Unique Departments: {filtered_df['Department'].nunique()}")
                if "Position" in filtered_df.columns:
                    st.write(f"Unique Positions: {filtered_df['Position'].nunique()}")

else:
    st.warning("⚠️ No data available. Please check your data source.")
    st.info("Upload a file using the sidebar or ensure the default file exists.")