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
    /* TRUE FULL SCREEN OVERRIDES */
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 98% !important; }
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
    
    /* KPI Grid */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
    .kpi-card { 
        background-color: #ffffff; border-radius: 10px; padding: 20px 15px; text-align: center; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.03); border: 1px solid #f0f0f0; transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 10px rgba(0,0,0,0.06); }
    .kpi-title { color: #888888; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .kpi-value { color: #0071CE; font-size: 28px; font-weight: 800; }
    
    /* Alerts */
    .critical-box { background-color: #fff0f1; border-left: 6px solid #DC3545; padding: 12px 20px; border-radius: 6px; margin-bottom: 15px; color: #721C24; font-weight: 500;}
    
    /* Content Cards */
    .content-card { background-color: #ffffff; border-radius: 10px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.02); border: 1px solid #f0f0f0; margin-bottom: 20px; height: 100%;}
    
    .takeaways-list { list-style: none; padding-left: 0; margin: 0; }
    .takeaways-list li { margin-bottom: 14px; padding-left: 24px; position: relative; font-size: 14.5px; color: #444; line-height: 1.5; }
    .takeaways-list li:before { content: '→'; position: absolute; left: 0; color: #0071CE; font-weight: bold; }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: 600; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

# --- 2. AUTHENTICATION LOGIC ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_password():
    if st.session_state['authenticated']:
        return True
        
    st.markdown("""
        <div class="login-container">
            <img src="https://logo.clearbit.com/walmart.com" width="70" style="margin-bottom: 15px; border-radius: 50%;">
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

# --- 5. DATA INGESTION & SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    sensitivity = st.slider("Rule Sensitivity", 0.01, 0.50, 0.05, 0.01)
    st.divider()
    uploaded_file = st.file_uploader("Override Transactions", type="csv")
    st.divider()
    if st.button("🔒 Secure Logout", use_container_width=True):
        st.session_state['authenticated'] = False
        st.rerun()

if uploaded_file is not None: df = pd.read_csv(uploaded_file)
elif os.path.exists("transactions_2000.csv"): df = pd.read_csv("transactions_2000.csv")
else: st.error("⚠️ 'transactions_2000.csv' not found."); st.stop()

total_txns = len(df)
all_items = [item.strip() for sublist in df['Items'].astype(str).str.split(',') for item in sublist]
avg_basket = round(len(all_items) / total_txns, 1) if total_txns > 0 else 0
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

# --- 6. MAIN DASHBOARD UI ---
st.markdown(f"""
<div class="main-banner">
    <h1><img src="https://logo.clearbit.com/walmart.com" width="40" style="border-radius: 50%;">&nbsp;Strategy Dashboard</h1>
    <div class="date-badge">{datetime.datetime.now().strftime("%B %Y")}</div>
</div>
""", unsafe_allow_html=True)

# 🎛️ CONTROLS
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

# --- 7. CORE ANALYTICS ---
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
        bar = base.mark_bar(color='#0071CE', cornerRadiusEnd=4, height=28)
        text = base.mark_text(align='left', dx=8, fontSize=12, fontWeight='bold', color='#0071CE').encode(
            text=alt.Text(f'{x_col}:Q', format='$.2f' if x_col == 'Pair_Profit' else '.1%')
        )
        st.altair_chart((bar + text).properties(height=220), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_takeaways:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Executive Takeaways")
        takeaways = ""
        for i, row in rules_df.head(3).iterrows():
            takeaways += f"<li>Buying <strong>{row['Item_A']}</strong> drives <strong>{row['Item_B']}</strong>. {row['Strategy'].split(':')[0]}</li>"
        st.markdown(f'<ul class="takeaways-list">{takeaways}</ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 8. MANAGER DIAGNOSTIC CENTER ---
    st.markdown("### 🔬 Manager Diagnostic Center")
    tab1, tab2, tab3 = st.tabs(["📉 Inventory Burn Risk", "🚀 Growth Opportunities", "👥 Traffic Analytics"])
    
    with tab1:
        st.markdown("Identify which products are at highest risk of stockouts based on pairing velocity.")
        inv_df = rules_df[['Item_A', 'Item_B', 'support']].copy()
        inv_df['Stock (A)'] = inv_df['Item_A'].apply(lambda x: PRODUCT_DB.get(x, {}).get('stock', 0))
        inv_df['Stock (B)'] = inv_df['Item_B'].apply(lambda x: PRODUCT_DB.get(x, {}).get('stock', 0))
        inv_df['Risk Level'] = inv_df.apply(lambda row: "🚨 High Risk" if row['Stock (B)'] < 30 else "✅ Stable", axis=1)
        st.dataframe(inv_df[['Item_A', 'Item_B', 'Stock (B)', 'Risk Level']], hide_index=True, use_container_width=True)
        
    with tab2:
        st.markdown("Products with high correlation (Lift) but low overall volume. Perfect targets for new endcap promotions.")
        growth_df = rules_df[(rules_df['lift'] > 1.5) & (rules_df['support'] < 0.15)].copy()
        if not growth_df.empty:
            growth_df['lift'] = growth_df['lift'].round(2)
            growth_df['support'] = (growth_df['support'] * 100).round(1).astype(str) + '%'
            st.dataframe(growth_df[['Item_A', 'Item_B', 'lift', 'support']], hide_index=True, use_container_width=True)
        else:
            st.info("Lower the Rule Sensitivity in the sidebar to reveal hidden growth opportunities.")
            
    with tab3:
        if not traffic_df.empty:
            traffic_chart = alt.Chart(traffic_df).mark_bar(color='#FFC220', size=40, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X('Day:N', title='', sort=cats, axis=alt.Axis(labelAngle=0, labelColor='#555')),
                y=alt.Y('Transactions:Q', axis=None)
            ).properties(height=250)
            st.altair_chart(traffic_chart, use_container_width=True)
else:
    st.info("ℹ️ No associations found. Adjust filters or lower sensitivity.")
