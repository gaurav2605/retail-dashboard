import streamlit as st
import pandas as pd
import altair as alt
import os
import datetime
import math
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Supercenter Operations", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* TRUE FULL SCREEN & HIDE SIDEBAR COMPLETELY */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 98% !important; }
    [data-testid="collapsedControl"] { display: none !important; } 
    header { visibility: hidden !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    body { font-family: 'Inter', 'Segoe UI', sans-serif; background-color: #f4f6f9; }
    
    /* Login Screen */
    .login-container { max-width: 420px; margin: 10vh auto; padding: 40px; background: white; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.08); border: 1px solid #eaeaea; text-align: center; }
    
    /* Main Banner */
    .main-banner {
        background-color: #0071CE; padding: 20px 30px; border-radius: 12px; margin-bottom: 20px; 
        color: white; box-shadow: 0 4px 15px rgba(0, 113, 206, 0.2);
        display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
    }
    .main-banner h1 { margin: 0; font-size: 28px; font-weight: 700; display: flex; align-items: center; gap: 15px;}
    .main-banner p { color: #FFC220; margin: 5px 0 0 0; font-size: 15px; font-weight: 500; }
    .date-badge { background-color: #FFC220; color: #004c8c; padding: 6px 20px; border-radius: 50px; font-weight: 800; font-size: 14px; }
    
    /* Executive Briefing Cards */
    .briefing-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 25px; }
    .briefing-card { padding: 15px 20px; border-radius: 8px; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.1); display: flex; flex-direction: column; justify-content: center;}
    .briefing-card h4 { margin: 0 0 8px 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.9;}
    .briefing-card p { margin: 0; font-size: 16px; font-weight: 600; line-height: 1.4;}
    
    /* KPI Grid */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
    .kpi-card { 
        background-color: #ffffff; border-radius: 10px; padding: 20px 15px; text-align: center; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.03); border: 1px solid #f0f0f0; transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.06); }
    .kpi-title { color: #888888; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .kpi-value { color: #0071CE; font-size: 30px; font-weight: 800; }
    
    /* Content Cards */
    .content-card { background-color: #ffffff; border-radius: 10px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); border: 1px solid #f0f0f0; margin-bottom: 20px; height: 100%;}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #e0e0e0;}
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

walmart_spark_svg = """<svg width="40" height="40" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M50 10 L50 32 M85 30 L67 45 M85 70 L67 55 M50 90 L50 68 M15 70 L33 55 M15 30 L33 45" stroke="#FFC220" stroke-width="10" stroke-linecap="round"/></svg>"""

# --- 2. AUTHENTICATION LOGIC ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_password():
    if st.session_state['authenticated']: return True
    st.markdown(f"""
        <div class="login-container">
            <div style="margin-bottom: 10px;">{walmart_spark_svg}</div>
            <h2 style="color: #333; margin-bottom: 5px;">Executive Portal</h2>
            <p style="color: #777; font-size: 14px; margin-bottom: 25px;">Secure diagnostic access required</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Secure Login", use_container_width=True)
            if submit_button:
                if username == "admin" and password == "manager123":
                    st.session_state['authenticated'] = True
                    st.rerun()
                else: st.error("❌ Invalid username or password.")
    return False

if not check_password(): st.stop()

# --- 3. INTERNAL DATABASE (Profit & Inventory) ---
PRODUCT_DB = {
    'Bread': {'profit': 0.75, 'stock': 12}, 'Butter': {'profit': 1.10, 'stock': 85},
    'Milk': {'profit': 0.50, 'stock': 150}, 'Diapers': {'profit': 6.00, 'stock': 200},
    'Beer': {'profit': 4.50, 'stock': 300}, 'Wine': {'profit': 12.00, 'stock': 40},
    'Cheese': {'profit': 4.00, 'stock': 90}, 'Eggs': {'profit': 0.80, 'stock': 60},
    'Coffee': {'profit': 3.50, 'stock': 110}, 'Nutella': {'profit': 2.50, 'stock': 5},
    'Snacks': {'profit': 2.00, 'stock': 500}
}

# --- 4. DATA ENGINE ---
def process_data(df, min_supp, total_txns):
    transactions = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    freq_items = apriori(df_encoded, min_support=min_supp, use_colnames=True)
    if freq_items.empty: return pd.DataFrame()
        
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
    if rules.empty: return pd.DataFrame()
    rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)].copy()
    if rules.empty: return pd.DataFrame()
        
    rules['Item_A'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Item_B'] = rules['consequents'].apply(lambda x: list(x)[0])
    rules['Profit_A'] = rules['Item_A'].apply(lambda x: PRODUCT_DB.get(x, {}).get('profit', 0))
    rules['Profit_B'] = rules['Item_B'].apply(lambda x: PRODUCT_DB.get(x, {}).get('profit', 0))
    rules['Pair_Profit'] = rules['Profit_A'] + rules['Profit_B']
    
    rules['Missed_Txns'] = ((rules['support'] / rules['confidence']) * total_txns) * (1 - rules['confidence'])
    rules['Missed_Profit'] = rules['Missed_Txns'] * rules['Profit_B']
    
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Item_A'], row['Item_B']]), axis=1)
    return rules.sort_values(by='confidence', ascending=False).drop_duplicates(subset=['Pair_ID'], keep='first')

# --- 5. ADMINISTRATOR CONTROLS ---
with st.expander("⚙️ System Configuration (Data & Engine Settings)"):
    admin_col1, admin_col2, admin_col3 = st.columns(3)
    with admin_col1: uploaded_file = st.file_uploader("Upload Point of Sale Data (CSV)", type="csv")
    with admin_col2: sensitivity = st.slider("Engine Sensitivity", 0.01, 0.50, 0.05, 0.01)
    with admin_col3: 
        st.write(""); st.write("")
        if st.button("🔒 Secure Logout", use_container_width=True):
            st.session_state['authenticated'] = False
            st.rerun()

# --- 6. DATA INGESTION ---
if uploaded_file is not None: df = pd.read_csv(uploaded_file)
elif os.path.exists("transactions_2000.csv"): df = pd.read_csv("transactions_2000.csv")
else: st.error("⚠️ 'transactions_2000.csv' not found."); st.stop()

df['Items_List'] = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])
df['Basket_Size'] = df['Items_List'].apply(len)

total_txns = len(df)
all_items = [item for sublist in df['Items_List'] for item in sublist]
avg_basket = round(df['Basket_Size'].mean(), 1) if total_txns > 0 else 0
best_seller = pd.Series(all_items).mode()[0] if all_items else "N/A"
total_revenue = sum([PRODUCT_DB.get(i, {}).get('profit', 1.00) for i in all_items])

busiest_day, traffic_df = "N/A", pd.DataFrame()
if 'Date' in df.columns:
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        busiest_day = df['Date'].dt.day_name().mode()[0]
        traffic_df = df['Date'].dt.day_name().value_counts().reset_index()
        traffic_df.columns = ['Day', 'Transactions']
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        traffic_df['Day'] = pd.Categorical(traffic_df['Day'], categories=cats, ordered=True)
        traffic_df = traffic_df.sort_values('Day')
        traffic_df['Est_Cashiers_Needed'] = (traffic_df['Transactions'] / 40).apply(math.ceil)
    except: pass

rules_df = process_data(df, sensitivity, total_txns)

# --- 7. EXECUTIVE DASHBOARD ---
st.markdown(f"""
<div class="main-banner">
    <div>
        <h1>{walmart_spark_svg} Advanced Operations Dashboard</h1>
        <p>Prescriptive Analytics & Profitability Engine</p>
    </div>
    <div class="date-badge">Live System Active • {datetime.datetime.now().strftime("%B %Y")}</div>
</div>
""", unsafe_allow_html=True)

# 🚀 CONSULTANT FEATURE 1: THE MORNING BRIEFING
st.markdown("### 📋 Today's Executive Priorities")
briefing_col1, briefing_col2, briefing_col3 = st.columns(3)

with briefing_col1:
    oos_item = "None"
    if not rules_df.empty:
        low_stock = rules_df.copy()
        low_stock['Stock'] = low_stock['Item_B'].apply(lambda x: PRODUCT_DB.get(x, {}).get('stock', 999))
        low_stock = low_stock[low_stock['Stock'] < 30].sort_values('Missed_Profit', ascending=False)
        if not low_stock.empty:
            oos_item = low_stock.iloc[0]['Item_B']
            risk_val = low_stock.iloc[0]['Missed_Profit']
            st.markdown(f'<div class="briefing-card" style="background: linear-gradient(135deg, #DC3545, #C82333);"><h4>🚨 Inventory Risk</h4><p>Restock <strong>{oos_item}</strong> immediately. You risk losing ${risk_val:,.0f} in bundled sales today.</p></div>', unsafe_allow_html=True)
        else: st.markdown(f'<div class="briefing-card" style="background: linear-gradient(135deg, #28A745, #218838);"><h4>✅ Inventory</h4><p>All highly-correlated products have sufficient stock.</p></div>', unsafe_allow_html=True)

with briefing_col2:
    if not rules_df.empty:
        top_bundle = rules_df.sort_values('Missed_Profit', ascending=False).iloc[0]
        st.markdown(f'<div class="briefing-card" style="background: linear-gradient(135deg, #0071CE, #0056b3);"><h4>📦 Planogram Action</h4><p>Build an endcap pairing <strong>{top_bundle["Item_A"]} & {top_bundle["Item_B"]}</strong>. Recapture ${top_bundle["Missed_Profit"]:,.0f} lost revenue.</p></div>', unsafe_allow_html=True)

with briefing_col3:
    if not traffic_df.empty:
        peak_day_row = traffic_df.loc[traffic_df['Transactions'].idxmax()]
        st.markdown(f'<div class="briefing-card" style="background: linear-gradient(135deg, #FFC220, #E0A800); color: #333;"><h4>👥 Labor Optimization</h4><p>Peak traffic expected on <strong>{peak_day_row["Day"]}</strong>. Ensure <strong>{peak_day_row["Est_Cashiers_Needed"]} cashiers</strong> are scheduled.</p></div>', unsafe_allow_html=True)

# KPIs
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">🧾 Txns Analyzed</div><div class="kpi-value">{total_txns:,}</div></div>
    <div class="kpi-card"><div class="kpi-title">🛍️ Avg Basket Size</div><div class="kpi-value">{avg_basket} <span style="font-size:14px; color:#888;">items</span></div></div>
    <div class="kpi-card"><div class="kpi-title">⭐ Best-Seller</div><div class="kpi-value">{best_seller}</div></div>
    <div class="kpi-card"><div class="kpi-title">💰 Est. Gross Margin</div><div class="kpi-value">${total_revenue:,.0f}</div></div>
</div>
""", unsafe_allow_html=True)

# --- 8. CONSULTANT DIAGNOSTIC CENTER ---
if not rules_df.empty:
    st.markdown("### 🔬 Consultant Diagnostic Center")
    tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Loss Leader Strategy", "👥 AI Labor Optimizer", "📈 Bundling Economics", "🛒 Trip Dynamics"])
    
    with tab1:
        st.markdown("**The Loss Leader Matrix:** Identifies low-margin items that act as 'bait' (high support/confidence) to drive the sales of highly profitable items. *Action: Discount the Driver, mark up the Partner.*")
        ll_df = rules_df[(rules_df['Profit_A'] < 1.50) & (rules_df['Profit_B'] > 3.00)].copy()
        
        if not ll_df.empty:
            ll_chart = alt.Chart(ll_df).mark_circle(size=400, opacity=0.9).encode(
                x=alt.X('confidence:Q', title='Probability of Buying High-Margin Item', axis=alt.Axis(format='%')),
                y=alt.Y('Profit_B:Q', title='Profit Gained on Partner Item ($)', axis=alt.Axis(format='$,.2f')),
                color=alt.Color('Item_A:N', title='Low-Margin Driver Item', scale=alt.Scale(scheme='category10')),
                tooltip=['Item_A', 'Profit_A', 'Item_B', 'Profit_B', 'confidence']
            ).properties(height=350)
            st.altair_chart(ll_chart, use_container_width=True)
        else:
            st.info("No clear loss-leader patterns detected at current sensitivity levels.")

    with tab2:
        st.markdown("**Labor vs. Traffic Optimizer:** Matches forecasted store traffic with required checkout headcount to prevent bottlenecks without overspending on payroll.")
        if not traffic_df.empty:
            base = alt.Chart(traffic_df).encode(x=alt.X('Day:N', sort=cats, axis=alt.Axis(labelAngle=0, labelColor='#555', labelFontSize=13)))
            bar = base.mark_bar(color='#E9ECEF', cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(y=alt.Y('Transactions:Q', title='Transaction Volume'))
            line = base.mark_line(color='#0071CE', strokeWidth=4).encode(y=alt.Y('Est_Cashiers_Needed:Q', title='Required Cashiers'))
            points = base.mark_circle(color='#FFC220', size=150).encode(y=alt.Y('Est_Cashiers_Needed:Q'), tooltip=['Day', 'Transactions', 'Est_Cashiers_Needed'])
            st.altair_chart(alt.layer(bar, line + points).resolve_scale(y='independent').properties(height=350), use_container_width=True)

    with tab3:
        st.markdown("**Planogram Optimization (Revenue Leakage):** Exact dollar amount lost when the layout fails to facilitate bundled purchases.")
        # FIX APPLIED HERE: labelFontWeight='bold' instead of fontWeight='bold'
        margin_chart = alt.Chart(rules_df.sort_values('Missed_Profit', ascending=False).head(6)).mark_bar(color='#28A745', cornerRadiusEnd=4, height=30).encode(
            x=alt.X('Missed_Profit:Q', title='Estimated Lost Profit ($)', axis=alt.Axis(format='$,.0f')),
            y=alt.Y('Pair_ID:N', sort='-x', title='Product Pairing', axis=alt.Axis(labelFontSize=13, labelFontWeight='bold')),
            tooltip=['Item_A', 'Item_B', 'Missed_Profit']
        ).properties(height=350)
        st.altair_chart(margin_chart, use_container_width=True)
            
    with tab4:
        st.markdown("**Customer Segmentation:** Understanding the mission of the shopper to adjust front-end merchandising.")
        quick_runs = df[df['Basket_Size'] <= 2].shape[0]
        stock_ups = df[df['Basket_Size'] >= 3].shape[0]
        trip_df = pd.DataFrame({"Trip Type": ["🏃‍♂️ Quick Run (1-2 items)", "🛒 Stock-Up (3+ items)"], "Total Transactions": [quick_runs, stock_ups]})
        
        donut = alt.Chart(trip_df).mark_arc(innerRadius=80, stroke='#fff', strokeWidth=2).encode(
            theta='Total Transactions:Q',
            color=alt.Color('Trip Type:N', scale=alt.Scale(range=['#0071CE', '#FFC220']), legend=alt.Legend(title="", orient="right", labelFontSize=15, symbolSize=300)),
            tooltip=['Trip Type', 'Total Transactions']
        ).properties(height=350)
        st.altair_chart(donut, use_container_width=True)
else:
    st.info("ℹ️ No strong associations found. Adjust filters in the System Configuration.")
