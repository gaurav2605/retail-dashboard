import streamlit as st
import pandas as pd
import altair as alt
import os
import datetime
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Supercenter Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* TRUE FULL SCREEN & HIDE SIDEBAR COMPLETELY */
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 98% !important; }
    [data-testid="collapsedControl"] { display: none !important; } /* Hides the sidebar expansion arrow */
    header { visibility: hidden !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    body { font-family: 'Inter', 'Segoe UI', sans-serif; background-color: #f4f6f9; }
    
    /* Login Screen */
    .login-container { max-width: 420px; margin: 10vh auto; padding: 40px; background: white; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.08); border: 1px solid #eaeaea; text-align: center; }
    
    /* Main Banner */
    .main-banner {
        background-color: #0071CE; padding: 25px 35px; border-radius: 12px; margin-bottom: 20px; 
        color: white; box-shadow: 0 4px 15px rgba(0, 113, 206, 0.2);
        display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
    }
    .main-banner h1 { margin: 0; font-size: 32px; font-weight: 700; display: flex; align-items: center; gap: 15px;}
    .main-banner p { color: #FFC220; margin: 5px 0 0 0; font-size: 16px; font-weight: 500; }
    .date-badge { background-color: #FFC220; color: #004c8c; padding: 6px 20px; border-radius: 50px; font-weight: 800; font-size: 14px; }
    
    /* KPI Grid */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
    .kpi-card { 
        background-color: #ffffff; border-radius: 10px; padding: 20px 15px; text-align: center; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.03); border: 1px solid #f0f0f0; transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.06); }
    .kpi-title { color: #888888; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .kpi-value { color: #0071CE; font-size: 32px; font-weight: 800; }
    
    /* Alerts */
    .critical-box { background-color: #fff0f1; border-left: 6px solid #DC3545; padding: 12px 20px; border-radius: 6px; margin-bottom: 15px; color: #721C24; font-weight: 500;}
    
    /* Content Cards */
    .content-card { background-color: #ffffff; border-radius: 10px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); border: 1px solid #f0f0f0; margin-bottom: 20px; height: 100%;}
    
    .takeaways-list { list-style: none; padding-left: 0; margin: 0; }
    .takeaways-list li { margin-bottom: 16px; padding-left: 24px; position: relative; font-size: 15px; color: #444; line-height: 1.5; }
    .takeaways-list li:before { content: '→'; position: absolute; left: 0; color: #0071CE; font-weight: bold; }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# RAW INLINE SVG LOGO (100% immune to browser blocking)
walmart_spark_svg = """
<svg width="45" height="45" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <path d="M50 10 L50 32 M85 30 L67 45 M85 70 L67 55 M50 90 L50 68 M15 70 L33 55 M15 30 L33 45" stroke="#FFC220" stroke-width="9" stroke-linecap="round"/>
</svg>
"""

# --- 2. AUTHENTICATION LOGIC ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_password():
    if st.session_state['authenticated']:
        return True
        
    st.markdown(f"""
        <div class="login-container">
            <div style="margin-bottom: 10px;">{walmart_spark_svg}</div>
            <h2 style="color: #333; margin-bottom: 5px;">Manager Portal</h2>
            <p style="color: #777; font-size: 14px; margin-bottom: 25px;">Secure diagnostic access required</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your User ID")
            password = st.text_input("Password", type="password", placeholder="Enter your Password")
            submit_button = st.form_submit_button("Secure Login", use_container_width=True)
            
            if submit_button:
                if username == "admin" and password == "manager123":
                    st.session_state['authenticated'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")
    return False

if not check_password():
    st.stop()


# --- 3. INTERNAL DATABASE (Profit & Inventory) ---
PRODUCT_DB = {
    'Bread': {'profit': 0.75, 'stock': 12}, 
    'Butter': {'profit': 1.10, 'stock': 85},
    'Milk': {'profit': 0.50, 'stock': 150},
    'Diapers': {'profit': 6.00, 'stock': 200},
    'Beer': {'profit': 4.50, 'stock': 300},
    'Wine': {'profit': 12.00, 'stock': 40},
    'Cheese': {'profit': 4.00, 'stock': 90},
    'Eggs': {'profit': 0.80, 'stock': 60},
    'Coffee': {'profit': 3.50, 'stock': 110},
    'Nutella': {'profit': 2.50, 'stock': 5},
    'Snacks': {'profit': 2.00, 'stock': 500}
}

# --- 4. HELPER FUNCTIONS ---
def generate_strategy(item_a, item_b, lift, conf, supp, missed_profit):
    if conf >= 0.95: return f"🏷️ DISCOUNT {item_a}: Drive sales of {item_b}. 100% of {item_a} buyers buy {item_b}."
    elif lift > 1.5 and conf >= 0.6: return f"📦 BUNDLE: Capture ${missed_profit:,.0f} in lost profit by bundling {item_b} with {item_a}."
    else: return f"🚶 SEPARATE: Place far apart to force store navigation."

def process_data(df, min_supp, optimize_for, total_txns):
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
    rules['Pair_Profit'] = rules.apply(lambda row: PRODUCT_DB.get(row['Item_A'], {}).get('profit', 0) + PRODUCT_DB.get(row['Item_B'], {}).get('profit', 0), axis=1)
    
    rules['Item_A_Txns'] = (rules['support'] / rules['confidence']) * total_txns
    rules['Missed_Txns'] = rules['Item_A_Txns'] * (1 - rules['confidence'])
    rules['Missed_Profit'] = rules.apply(lambda row: row['Missed_Txns'] * PRODUCT_DB.get(row['Item_B'], {}).get('profit', 0), axis=1)
    
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Item_A'], row['Item_B']]), axis=1)
    rules = rules.sort_values(by='confidence', ascending=False).drop_duplicates(subset=['Pair_ID'], keep='first')
    
    if optimize_for == "💰 Maximum Profit": rules = rules.sort_values(by='Pair_Profit', ascending=False)
    else: rules = rules.sort_values(by='support', ascending=False)
        
    rules['Strategy'] = rules.apply(lambda row: generate_strategy(row['Item_A'], row['Item_B'], row['lift'], row['confidence'], row['support'], row['Missed_Profit']), axis=1)
    return rules

# --- 5. ADMINISTRATOR CONTROLS (Replacing Sidebar) ---
with st.expander("⚙️ Administrator Tools (Data Upload & Settings)"):
    st.markdown("Use this panel to manage system data or log out of the terminal.")
    admin_col1, admin_col2, admin_col3 = st.columns(3)
    
    with admin_col1:
        uploaded_file = st.file_uploader("Override Transactions CSV", type="csv")
    with admin_col2:
        sensitivity = st.slider("Engine Rule Sensitivity", 0.01, 0.50, 0.05, 0.01)
        st.caption("Lower threshold = rarer patterns.")
    with admin_col3:
        st.write("")
        st.write("")
        if st.button("🔒 Secure Logout", use_container_width=True):
            st.session_state['authenticated'] = False
            st.rerun()

# --- 6. DATA INGESTION ---
if uploaded_file is not None: df = pd.read_csv(uploaded_file)
elif os.path.exists("transactions_2000.csv"): df = pd.read_csv("transactions_2000.csv")
else: st.error("⚠️ 'transactions_2000.csv' not found. Please upload a file via the Admin Tools above."); st.stop()

df['Items_List'] = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])
df['Basket_Size'] = df['Items_List'].apply(len)

total_txns = len(df)
all_items = [item for sublist in df['Items_List'] for item in sublist]
avg_basket = round(df['Basket_Size'].mean(), 1) if total_txns > 0 else 0
best_seller = pd.Series(all_items).mode()[0] if all_items else "N/A"
total_revenue = sum([PRODUCT_DB.get(i, {}).get('profit', 1.00) for i in all_items])

busiest_day = "N/A"
traffic_df = pd.DataFrame()
if 'Date' in df.columns:
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        busiest_day = df['Date'].dt.day_name().mode()[0]
        traffic_df = df['Date'].dt.day_name().value_counts().reset_index()
        traffic_df.columns = ['Day', 'Transactions']
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        traffic_df['Day'] = pd.Categorical(traffic_df['Day'], categories=cats, ordered=True)
        traffic_df = traffic_df.sort_values('Day')
    except: pass

# --- 7. MAIN DASHBOARD UI ---
st.markdown(f"""
<div class="main-banner">
    <div>
        <h1>{walmart_spark_svg} Strategy Dashboard</h1>
        <p>Data-driven layout and pricing intelligence</p>
    </div>
    <div class="date-badge">{datetime.datetime.now().strftime("%B %Y")}</div>
</div>
""", unsafe_allow_html=True)

# 🎛️ IN-PAGE CONTROLS
control_col1, control_col2 = st.columns(2)
with control_col1:
    optimization = st.radio("Optimize layout for:", ["📦 Sales Volume", "💰 Maximum Profit"], horizontal=True, label_visibility="collapsed")
with control_col2:
    target_product = st.selectbox("Targeted Inventory Filter:", ["Show All Products"] + sorted(list(set(all_items))), label_visibility="collapsed")

rules_df = process_data(df, sensitivity, optimization, total_txns)
if target_product != "Show All Products" and not rules_df.empty:
    rules_df = rules_df[(rules_df['Item_A'] == target_product) | (rules_df['Item_B'] == target_product)]

# Alerts
if not rules_df.empty:
    for i, row in rules_df.head(2).iterrows():
        if PRODUCT_DB.get(row['Item_B'], {}).get('stock', 999) < 20:
            st.markdown(f'<div class="critical-box">⚠️ <strong>INVENTORY CRITICAL:</strong> Only {PRODUCT_DB.get(row["Item_B"], {}).get("stock")} units of {row["Item_B"]} remaining. Restock to support {row["Item_A"]} sales.</div>', unsafe_allow_html=True)

# KPIs
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">🧾 Txns Analyzed</div><div class="kpi-value">{total_txns:,}</div></div>
    <div class="kpi-card"><div class="kpi-title">🛍️ Avg Basket</div><div class="kpi-value">{avg_basket} <span style="font-size:14px; color:#888;">items</span></div></div>
    <div class="kpi-card"><div class="kpi-title">⭐ Best-Seller</div><div class="kpi-value">{best_seller}</div></div>
    <div class="kpi-card"><div class="kpi-title">📅 Peak Day</div><div class="kpi-value">{busiest_day}</div></div>
    <div class="kpi-card"><div class="kpi-title">💰 Est. Profit</div><div class="kpi-value">${total_revenue:,.0f}</div></div>
</div>
""", unsafe_allow_html=True)

# --- 8. CORE ANALYTICS ---
if not rules_df.empty:
    col_chart, col_takeaways = st.columns([1.5, 1])
    
    with col_chart:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        metric_name = "Volume" if optimization == "📦 Sales Volume" else "Profit"
        st.markdown(f"#### 📊 Top 5 Pairings <span style='font-size: 13px; color: #888;'>(by {metric_name})</span>", unsafe_allow_html=True)
        
        chart_data = rules_df.head(5).copy()
        chart_data['Pair'] = chart_data['Item_A'] + " & " + chart_data['Item_B']
        x_col = 'Pair_Profit' if optimization == "💰 Maximum Profit" else 'support'
        
        base = alt.Chart(chart_data).encode(
            x=alt.X(f'{x_col}:Q', axis=None), y=alt.Y('Pair:N', sort='-x', title='', axis=alt.Axis(labelFontSize=13, tickSize=0, domain=False))
        )
        bar = base.mark_bar(color='#0071CE', cornerRadiusEnd=4, height=35)
        text = base.mark_text(align='left', dx=8, fontSize=13, fontWeight='bold', color='#0071CE').encode(
            text=alt.Text(f'{x_col}:Q', format='$.2f' if x_col == 'Pair_Profit' else '.1%')
        )
        st.altair_chart((bar + text).properties(height=260), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_takeaways:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Executive Takeaways")
        takeaways = ""
        for i, row in rules_df.head(3).iterrows():
            takeaways += f"<li>Buying <strong>{row['Item_A']}</strong> drives <strong>{row['Item_B']}</strong>. {row['Strategy'].split(':')[0]}</li>"
        st.markdown(f'<ul class="takeaways-list">{takeaways}</ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 9. ACTION PLAN ---
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Intelligent Action Plan")
    display_cols = rules_df[['Item_A', 'Item_B', 'Strategy']].copy()
    display_cols.columns = ['Driver Product', 'Partner Product', 'Recommended Execution']
    st.dataframe(display_cols, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 10. MANAGER DIAGNOSTIC CENTER (Expanded Tabs) ---
    st.markdown("### 🔬 Manager Diagnostic Center")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📉 Inventory Burn Risk", "🚀 Growth Opportunities", "👥 Traffic Analytics", "🛒 Trip Type Analysis", "📈 Margin Analyzer"])
    
    with tab1:
        st.markdown("**Identify which products are at highest risk of stockouts based on pairing velocity.**")
        inv_df = rules_df[['Item_A', 'Item_B', 'support']].copy()
        inv_df['Stock (A)'] = inv_df['Item_A'].apply(lambda x: PRODUCT_DB.get(x, {}).get('stock', 0))
        inv_df['Stock (B)'] = inv_df['Item_B'].apply(lambda x: PRODUCT_DB.get(x, {}).get('stock', 0))
        inv_df['Risk Level'] = inv_df.apply(lambda row: "🚨 High Risk" if row['Stock (B)'] < 30 else "✅ Stable", axis=1)
        st.dataframe(inv_df[['Item_A', 'Item_B', 'Stock (B)', 'Risk Level']], hide_index=True, use_container_width=True)
        
    with tab2:
        st.markdown("**Products with high correlation (Lift) but low overall volume. Perfect targets for new endcap promotions.**")
        growth_df = rules_df[(rules_df['lift'] > 1.5) & (rules_df['support'] < 0.15)].copy()
        if not growth_df.empty:
            growth_df['lift'] = growth_df['lift'].round(2)
            growth_df['support'] = (growth_df['support'] * 100).round(1).astype(str) + '%'
            st.dataframe(growth_df[['Item_A', 'Item_B', 'lift', 'support']], hide_index=True, use_container_width=True)
        else:
            st.info("Lower the Rule Sensitivity in the Administrator Tools to reveal hidden growth opportunities.")
            
    with tab3:
        st.markdown("**Store Traffic Heatmap to optimize staff scheduling.**")
        if not traffic_df.empty:
            traffic_chart = alt.Chart(traffic_df).mark_bar(color='#FFC220', size=40, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X('Day:N', title='', sort=cats, axis=alt.Axis(labelAngle=0, labelColor='#555')),
                y=alt.Y('Transactions:Q', axis=None)
            ).properties(height=280)
            st.altair_chart(traffic_chart, use_container_width=True)
            
    with tab4:
        st.markdown("**Segment checkouts to understand shopping behavior: Quick Runs vs. Stock-Up Trips.**")
        quick_runs = df[df['Basket_Size'] <= 2].shape[0]
        stock_ups = df[df['Basket_Size'] >= 3].shape[0]
        
        trip_df = pd.DataFrame({
            "Trip Type": ["🏃‍♂️ Quick Run (1-2 items)", "🛒 Stock-Up (3+ items)"],
            "Total Transactions": [quick_runs, stock_ups],
            "% of Store Traffic": [f"{(quick_runs/total_txns)*100:.1f}%", f"{(stock_ups/total_txns)*100:.1f}%"]
        })
        st.dataframe(trip_df, hide_index=True, use_container_width=True)
        
    with tab5:
        st.markdown("**Money Left on the Table: Calculate exact revenue lost when customers fail to bundle.**")
        analyst_df = rules_df[['Item_A', 'Item_B', 'confidence', 'Missed_Txns', 'Missed_Profit']].copy()
        analyst_df['confidence'] = (analyst_df['confidence'] * 100).round(1).astype(str) + '%'
        analyst_df['Missed_Txns'] = analyst_df['Missed_Txns'].round(0).astype(int)
        analyst_df['Missed_Profit'] = analyst_df['Missed_Profit'].apply(lambda x: f"${x:,.2f}")
        analyst_df.columns = ['Driver (A)', 'Partner (B)', 'Checkout Probability', 'Missed Checkouts', 'Missed Profit ($)']
        st.dataframe(analyst_df, hide_index=True, use_container_width=True)
else:
    st.info("ℹ️ No associations found. Adjust filters or lower sensitivity in the Administrator Tools.")
