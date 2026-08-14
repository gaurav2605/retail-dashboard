import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="AI Diagnostic Analytics", layout="wide", page_icon="📊")

# Custom CSS for Professional UI
st.markdown("""
    <style>
    .main { background-color: #F8FAFC; }
    .stApp { font-family: 'Inter', sans-serif; }
    .kpi-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        border-left: 5px solid #4F46E5; /* Indigo accent */
        margin-bottom: 20px;
    }
    .kpi-title { color: #64748B; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
    .kpi-value { color: #0F172A; font-size: 1.8rem; font-weight: 700; margin: 5px 0; }
    .kpi-change-pos { color: #10B981; font-weight: 600; font-size: 0.9rem;}
    .kpi-change-neg { color: #EF4444; font-weight: 600; font-size: 0.9rem;}
    .insight-box {
        background-color: #EEF2FF;
        border: 1px solid #C7D2FE;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #312E81;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("diagnostic_ecommerce_1500.xlsx")
        
        # Date conversions
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
        # Boolean conversions for rates (assuming Yes/No or True/False strings)
        if df['Returned'].dtype == 'O':
            df['Returned_Num'] = df['Returned'].astype(str).str.lower().map({'yes': 1, 'true': 1, 'no': 0, 'false': 0}).fillna(0)
        else:
            df['Returned_Num'] = df['Returned']
            
        if df['Complaint'].dtype == 'O':
            df['Complaint_Num'] = df['Complaint'].astype(str).str.lower().map({'yes': 1, 'true': 1, 'no': 0, 'false': 0}).fillna(0)
        else:
            df['Complaint_Num'] = df['Complaint']
            
        # ROI calculation
        df['ROI'] = (df['Revenue'] - df['Marketing_Spend']) / df['Marketing_Spend'].replace(0, np.nan)
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.stop()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Executive Overview", 
    "Diagnostic Analysis", 
    "Customer Analysis", 
    "Product & Operations", 
    "Marketing Analysis"
])

# ==========================================
# GLOBAL FILTERS
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("Global Filters")

if st.sidebar.button("Reset Filters"):
    st.experimental_rerun()

# Date Filter
min_date = df_raw['Order_Date'].min().date()
max_date = df_raw['Order_Date'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Other Filters
region_filter = st.sidebar.multiselect("Region", df_raw['Region'].unique())
category_filter = st.sidebar.multiselect("Category", df_raw['Category'].unique())
segment_filter = st.sidebar.multiselect("Customer Segment", df_raw['Customer_Segment'].unique())
channel_filter = st.sidebar.multiselect("Marketing Channel", df_raw['Marketing_Channel'].unique())

# Apply Filters
df = df_raw[(df_raw['Order_Date'].dt.date >= start_date) & (df_raw['Order_Date'].dt.date <= end_date)]
if region_filter: df = df[df['Region'].isin(region_filter)]
if category_filter: df = df[df['Category'].isin(category_filter)]
if segment_filter: df = df[df['Customer_Segment'].isin(segment_filter)]
if channel_filter: df = df[df['Marketing_Channel'].isin(channel_filter)]

# Helper function for currency formatting
def format_inr(value):
    return f"₹{value:,.0f}"

# ==========================================
# HELPER COMPONENTS
# ==========================================
def render_kpi_card(title, value, is_currency=False, is_percent=False):
    if is_currency: formatted_val = format_inr(value)
    elif is_percent: formatted_val = f"{value:.1f}%"
    else: formatted_val = f"{value:,.0f}"
    
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{formatted_val}</div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "Executive Overview":
    st.title("Executive Overview")
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: render_kpi_card("Total Revenue", df['Revenue'].sum(), is_currency=True)
    with c2: render_kpi_card("Total Profit", df['Profit'].sum(), is_currency=True)
    with c3: render_kpi_card("Total Orders", df['Order_ID'].nunique())
    with c4: render_kpi_card("Profit Margin", (df['Profit'].sum() / df['Revenue'].sum()) * 100 if df['Revenue'].sum() > 0 else 0, is_percent=True)
    
    c5, c6, c7, c8 = st.columns(4)
    with c5: render_kpi_card("Avg Order Value", df['Revenue'].sum() / df['Order_ID'].nunique(), is_currency=True)
    with c6: render_kpi_card("Return Rate", df['Returned_Num'].mean() * 100, is_percent=True)
    with c7: render_kpi_card("Complaint Rate", df['Complaint_Num'].mean() * 100, is_percent=True)
    with c8: render_kpi_card("Avg Rating", df['Customer_Rating'].mean() or 0)

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        rev_time = df.groupby(df['Order_Date'].dt.to_period("M"))['Revenue'].sum().reset_index()
        rev_time['Order_Date'] = rev_time['Order_Date'].dt.to_timestamp()
        fig_rev = px.line(rev_time, x='Order_Date', y='Revenue', title="Revenue Over Time", color_discrete_sequence=['#4F46E5'])
        st.plotly_chart(fig_rev, use_container_width=True)
        
    with col2:
        prof_time = df.groupby(df['Order_Date'].dt.to_period("M"))['Profit'].sum().reset_index()
        prof_time['Order_Date'] = prof_time['Order_Date'].dt.to_timestamp()
        fig_prof = px.line(prof_time, x='Order_Date', y='Profit', title="Profit Over Time", color_discrete_sequence=['#10B981'])
        st.plotly_chart(fig_prof, use_container_width=True)

    col3, col4, col5 = st.columns(3)
    with col3:
        cat_rev = df.groupby('Category')['Revenue'].sum().reset_index()
        st.plotly_chart(px.bar(cat_rev, x='Category', y='Revenue', title="Revenue by Category"), use_container_width=True)
    with col4:
        reg_rev = df.groupby('Region')['Revenue'].sum().reset_index()
        st.plotly_chart(px.pie(reg_rev, names='Region', values='Revenue', title="Revenue by Region", hole=0.4), use_container_width=True)
    with col5:
        seg_ord = df.groupby('Customer_Segment')['Order_ID'].nunique().reset_index()
        st.plotly_chart(px.bar(seg_ord, x='Customer_Segment', y='Order_ID', title="Orders by Segment"), use_container_width=True)

# ==========================================
# PAGE 2: DIAGNOSTIC ANALYSIS (The Core)
# ==========================================
elif page == "Diagnostic Analysis":
    st.title("Diagnostic Analysis")
    
    kpi_map = {
        "Revenue": {"col": "Revenue", "agg": "sum", "type": "currency", "good": "up"},
        "Profit": {"col": "Profit", "agg": "sum", "type": "currency", "good": "up"},
        "Orders": {"col": "Order_ID", "agg": "nunique", "type": "number", "good": "up"},
        "Return Rate": {"col": "Returned_Num", "agg": "mean", "type": "percent", "good": "down"},
        "Complaint Rate": {"col": "Complaint_Num", "agg": "mean", "type": "percent", "good": "down"},
        "Customer Rating": {"col": "Customer_Rating", "agg": "mean", "type": "number", "good": "up"}
    }
    
    selected_kpi = st.selectbox("Select KPI to Analyze:", list(kpi_map.keys()))
    kpi_info = kpi_map[selected_kpi]
    
    # Period Calculation (Current vs Previous half of selected date range)
    total_days = (end_date - start_date).days
    if total_days < 2:
        st.warning("Please select a wider date range (at least 2 days) to enable comparative diagnostic analysis.")
    else:
        mid_date = start_date + timedelta(days=total_days//2)
        
        df_prev = df[df['Order_Date'].dt.date < mid_date]
        df_curr = df[df['Order_Date'].dt.date >= mid_date]
        
        def calc_kpi(data, info):
            if data.empty: return 0
            if info['agg'] == 'sum': return data[info['col']].sum()
            elif info['agg'] == 'mean': return data[info['col']].mean()
            elif info['agg'] == 'nunique': return data[info['col']].nunique()
            
        val_curr = calc_kpi(df_curr, kpi_info)
        val_prev = calc_kpi(df_prev, kpi_info)
        
        pct_change = ((val_curr - val_prev) / val_prev * 100) if val_prev != 0 else 0
        
        # Display Comparative KPI
        c1, c2, c3 = st.columns(3)
        fmt = format_inr if kpi_info['type'] == 'currency' else (lambda x: f"{x*100:.1f}%" if kpi_info['type']=='percent' else lambda x: f"{x:,.2f}")
        
        with c1: st.metric("Current Period", fmt(val_curr))
        with c2: st.metric("Previous Period", fmt(val_prev))
        with c3: st.metric("% Change", f"{pct_change:+.1f}%")

        # Dimension Analysis
        st.subheader("Diagnostic Drill-Down")
        
        # Tree Map for Contribution
        if kpi_info['agg'] == 'sum':
            fig_tree = px.treemap(df, path=[px.Constant("All"), 'Category', 'Region', 'Customer_Segment'], 
                                  values=kpi_info['col'], title=f"{selected_kpi} Contribution Breakdown")
            st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info(f"Treemap drill-down is reserved for cumulative metrics (Revenue, Profit).")

        # AI Diagnostic Insight Generation Engine
        st.subheader("AI Diagnostic Insights")
        
        # Find best/worst contributors
        def get_dim_change(dim):
            curr = df_curr.groupby(dim).apply(lambda x: calc_kpi(x, kpi_info))
            prev = df_prev.groupby(dim).apply(lambda x: calc_kpi(x, kpi_info))
            diff = curr.sub(prev, fill_value=0)
            return diff.idxmax(), diff.max(), diff.idxmin(), diff.min()
            
        best_cat, best_cat_val, worst_cat, worst_cat_val = get_dim_change('Category')
        best_reg, best_reg_val, worst_reg, worst_reg_val = get_dim_change('Region')
        
        # Driver Correlations
        numeric_cols = ['Discount_%', 'Delivery_Days', 'Marketing_Spend', 'Competitor_Price', 'Stock_Availability']
        correlations = df[numeric_cols + [kpi_info['col']]].corr()[kpi_info['col']].drop(kpi_info['col'])
        
        # Formulate Insights
        trend_word = "increased" if pct_change > 0 else "decreased"
        
        key_findings = f"• **{selected_kpi}** {trend_word} by **{abs(pct_change):.1f}%** in the current period compared to the previous period.<br>"
        key_findings += f"• **{best_cat}** was the strongest positive contributor, whereas **{worst_cat}** saw the most significant negative pressure.<br>"
        key_findings += f"• Regionally, **{best_reg}** outperformed, while **{worst_reg}** lagged behind."

        drivers = ""
        strong_corr = correlations[abs(correlations) > 0.15]
        if not strong_corr.empty:
            for feat, val in strong_corr.items():
                direction = "higher" if val > 0 else "lower"
                drivers += f"• There is a notable correlation ({val:.2f}) between **{feat}** and {selected_kpi}. {direction.title()} {feat} is associated with increased {selected_kpi}.<br>"
        else:
            drivers = "• No strong linear correlations detected between standard operational metrics (Delivery, Discount, etc.) and this KPI during the selected period."

        # Render AI Box
        st.markdown(f"""
        <div class="insight-box">
            <h4>🧠 Key Findings</h4>
            <p>{key_findings}</p>
            <h4>🔍 Possible Drivers</h4>
            <p>{drivers}</p>
            <h4>🎯 Recommended Actions</h4>
            <p>• Investigate the root cause of the drop in {worst_cat} category within the {worst_reg} region.<br>
            • Consider replicating the successful operational practices seen in {best_cat} and {best_reg}.<br>
            • Monitor identified drivers closely to mitigate risks before they impact the bottom line.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 3: CUSTOMER ANALYSIS
# ==========================================
elif page == "Customer Analysis":
    st.title("Customer Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names='Customer_Segment', values='Revenue', title="Revenue by Segment", hole=0.3), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(df.groupby('Age_Group')['Order_ID'].nunique().reset_index(), x='Age_Group', y='Order_ID', title="Orders by Age Group"), use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        ret_seg = df.groupby('Customer_Segment')['Returned_Num'].mean().reset_index()
        st.plotly_chart(px.bar(ret_seg, x='Customer_Segment', y='Returned_Num', title="Return Rate by Segment", color='Returned_Num', color_continuous_scale='Reds'), use_container_width=True)
    with col4:
        comp_seg = df.groupby('Customer_Segment')['Complaint_Num'].mean().reset_index()
        st.plotly_chart(px.bar(comp_seg, x='Customer_Segment', y='Complaint_Num', title="Complaint Rate by Segment", color='Complaint_Num', color_continuous_scale='Oranges'), use_container_width=True)

# ==========================================
# PAGE 4: PRODUCT & OPERATIONS
# ==========================================
elif page == "Product & Operations":
    st.title("Product & Operations")
    
    # Top/Bottom Products
    st.subheader("Product Performance")
    prod_perf = df.groupby('Product').agg({'Revenue': 'sum', 'Profit': 'sum', 'Returned_Num':'mean'}).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        top_10 = prod_perf.nlargest(10, 'Profit')
        st.plotly_chart(px.bar(top_10, x='Profit', y='Product', orientation='h', title="Top 10 Products (Profit)"), use_container_width=True)
    with col2:
        bottom_10 = prod_perf.nsmallest(10, 'Profit')
        st.plotly_chart(px.bar(bottom_10, x='Profit', y='Product', orientation='h', title="Bottom 10 Products (Profit)", color_discrete_sequence=['#EF4444']), use_container_width=True)

    # Operations scatter
    st.subheader("Operational Drivers")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(px.scatter(df, x='Discount_%', y='Profit', color='Category', title="Discount vs Profit", trendline="ols"), use_container_width=True)
    with c4:
        st.plotly_chart(px.scatter(df, x='Delivery_Days', y='Customer_Rating', color='Region', title="Delivery Days vs Rating", trendline="ols"), use_container_width=True)

# ==========================================
# PAGE 5: MARKETING ANALYSIS
# ==========================================
elif page == "Marketing Analysis":
    st.title("Marketing Analysis")
    
    mkt_summary = df.groupby('Marketing_Channel').agg(
        Revenue=('Revenue', 'sum'),
        Spend=('Marketing_Spend', 'sum'),
        Orders=('Order_ID', 'nunique')
    ).reset_index()
    mkt_summary['ROI'] = (mkt_summary['Revenue'] - mkt_summary['Spend']) / mkt_summary['Spend']
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(mkt_summary, x='Marketing_Channel', y=['Revenue', 'Spend'], barmode='group', title="Revenue vs Spend by Channel"), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(mkt_summary, x='Marketing_Channel', y='ROI', title="ROI by Channel", color='ROI', color_continuous_scale='RdYlGn'), use_container_width=True)
        
    st.subheader("Campaign Performance")
    camp_summary = df.groupby('Campaign').agg({'Revenue':'sum', 'Profit':'sum', 'Order_ID':'nunique'}).reset_index()
    st.dataframe(camp_summary.style.format({'Revenue': '₹{:,.0f}', 'Profit': '₹{:,.0f}'}), use_container_width=True)
