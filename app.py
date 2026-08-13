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
    body { font-family: 'Inter', 'Segoe UI', sans-serif; background-color: #f8f9fa; }
    
    /* Login Screen */
    .login-container { max-width: 400px; margin: 80px auto; padding: 40px; background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.05); border: 1px solid #eaeaea; text-align: center; }
    .login-logo { margin-bottom: 20px; }
    
    .main-banner {
        background-color: #0071CE; padding: 25px 35px; border-radius: 12px; margin-bottom: 25px; 
        color: white; box-shadow: 0 4px 15px rgba(0, 113, 206, 0.2);
        display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
    }
    .main-banner h1 { margin: 0; font-size: 30px; font-weight: 700; letter-spacing: -0.5px; }
    .main-banner p { color: #FFC220; margin: 5px 0 0 0; font-size: 16px; font-weight: 500; }
    .date-badge { background-color: #FFC220; color: #004c8c; padding: 8px 24px; border-radius: 50px; font-weight: 800; font-size: 15px; }
    
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 30px; }
    .kpi-card { 
        background-color: #ffffff; border-radius: 10px; padding: 25px 20px; text-align: center; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.04); border: 1px solid #f0f0f0; transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .kpi-title { color: #888888; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .kpi-value { color: #0071CE; font-size: 32px; font-weight: 800; }
    
    .alert-box { background-color: #fff9e6; border-left: 6px solid #FFC107; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; color: #856404; font-weight: 500;}
    .critical-box { background-color: #fff0f1; border-left: 6px solid #DC3545; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; color: #721C24; font-weight: 500;}
    
    .content-card { background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #f0f0f0; margin-bottom: 30px; height: 100%;}
    
    .takeaways-list { list-style: none; padding-left: 0; margin: 0; }
    .takeaways-list li { margin-bottom: 16px; padding-left: 24px; position: relative; font-size: 15px; color: #444; line-height: 1.5; }
    .takeaways-list li:before { content: '→'; position: absolute; left: 0; color: #0071CE; font-weight: bold; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
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
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/512px-Walmart_logo.svg.png" width="150" class="login-logo">
            <h2 style="color: #333; margin-bottom: 5px;">Manager Portal</h2>
            <p style="color: #777; font-size: 14px; margin-bottom: 25px;">Secure access required</p>
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

# Halt the app here if the user is not logged in
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
    conf_pct = round(conf * 100)
    if conf >= 0.95:
        return f"🏷️ DISCOUNT {item_a}: Drive sales of {item_b}. 100% of {item_a} buyers buy {item_b}."
    elif lift > 1.5 and conf >= 0.6:
        return f"📦 BUNDLE: Capture ${missed_profit:,.0f} in lost profit by putting {item_b} next to {item_a}."
    else:
        return f"🚶 SEPARATE: Place far apart. Appears in {round(supp*100)}% of carts; forces store navigation."

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
    
    if optimize_for == "💰 Maximum Profit":
        rules = rules.sort_values(by='Pair_Profit', ascending=False)
    else:
        rules = rules.sort_values(by='support', ascending=False)
        
    rules['Strategy'] = rules.apply(lambda row: generate_strategy(row['Item_A'], row['Item_B'], row['lift'], row['confidence'], row['support'], row['Missed_Profit']), axis=1)
    return rules

# --- 5. DATA INGESTION & SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    sensitivity = st.slider("Rule Sensitivity", 0.01, 0.50, 0.05, 0.01)
    st.caption("Lowering this threshold finds rarer patterns.")
    st.divider()
    st.markdown("### 📥 Manual Upload")
    uploaded_file = st.file_uploader("Override Transactions", type="csv")
    st.divider()
    if st.button("🔒 Secure Logout", use_container_width=True):
        st.session_state['authenticated'] = False
        st.rerun()

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists("transactions_2000.csv"):
    df = pd.read_csv("transactions_2000.csv")
else:
    st.error("⚠️ 'transactions_2000.csv' not found. Please upload a file to continue.")
    st.stop()

# Basic Metrics
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
dynamic_month = datetime.datetime.now().strftime("%B %Y")

st.markdown(f"""
<div class="main-banner">
    <div>
        <h1>Strategy Dashboard</h1>
        <p>Data-driven layout and pricing intelligence</p>
    </div>
    <div class="date-badge">{dynamic_month}</div>
</div>
""", unsafe_allow_html=True)

# 🎛️ IN-PAGE CONTROLS
st.markdown("### 🎛️ Dashboard Controls")
control_col1, control_col2 = st.columns(2)
with control_col1:
    optimization = st.radio("Optimize floor layout for:", ["📦 Sales Volume", "💰 Maximum Profit"], horizontal=True)
with control_col2:
    target_product = st.selectbox("Targeted Inventory Filter (Focus on a specific product):", ["Show All Products"] + sorted(list(set(all_items))))
st.divider()

rules_df = process_data(df, sensitivity, optimization, total_txns)
if target_product != "Show All Products" and not rules_df.empty:
    rules_df = rules_df[(rules_df['Item_A'] == target_product) | (rules_df['Item_B'] == target_product)]

# Alerts
if not rules_df.empty:
    for i, row in rules_df.head(3).iterrows():
        stock_b = PRODUCT_DB.get(row['Item_B'], {}).get('stock', 999)
        if stock_b < 20:
            st.markdown(f'<div class="critical-box">⚠️ <strong>INVENTORY CRITICAL:</strong> Only {stock_b} units of {row["Item_B"]} remaining. Restock immediately to capture guaranteed follow-on sales from {row["Item_A"]}.</div>', unsafe_allow_html=True)

# KPI Row
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">🧾 Transactions Analyzed</div><div class="kpi-value">{total_txns:,}</div></div>
    <div class="kpi-card"><div class="kpi-title">🛍️ Avg. Basket Size</div><div class="kpi-value">{avg_basket} <span style="font-size:16px; font-weight:600; color:#888;">items</span></div></div>
    <div class="kpi-card"><div class="kpi-title">⭐ Best-Selling Product</div><div class="kpi-value">{best_seller}</div></div>
    <div class="kpi-card"><div class="kpi-title">📅 Busiest Store Day</div><div class="kpi-value">{busiest_day}</div></div>
    <div class="kpi-card"><div class="kpi-title">💰 Est. Total Profit</div><div class="kpi-value">${total_revenue:,.2f}</div></div>
</div>
""", unsafe_allow_html=True)

# --- 7. ADVANCED INSIGHTS (Charts) ---
if not rules_df.empty:
    col_chart, col_traffic = st.columns(2)
    
    with col_chart:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        chart_title_metric = "Volume" if optimization == "📦 Sales Volume" else "Profit"
        st.markdown(f"#### 📊 Top 5 Product Pairings <span style='font-size: 14px; color: #888; font-weight: normal;'>(Ranked by {chart_title_metric})</span>", unsafe_allow_html=True)
        
        chart_data = rules_df.head(5).copy()
        chart_data['Pair Name'] = chart_data['Item_A'] + " & " + chart_data['Item_B']
        x_col = 'Pair_Profit' if optimization == "💰 Maximum Profit" else 'support'
        
        base = alt.Chart(chart_data).encode(
            x=alt.X(f'{x_col}:Q', axis=None),
            y=alt.Y('Pair Name:N', sort='-x', title='', axis=alt.Axis(labelFontSize=13, labelColor='#555', tickSize=0, domain=False))
        )
        bar = base.mark_bar(color='#0071CE', cornerRadiusEnd=4, height=32)
        text = base.mark_text(align='left', baseline='middle', dx=8, fontSize=13, fontWeight='bold', color='#0071CE').encode(
            text=alt.Text(f'{x_col}:Q', format='$.2f' if x_col == 'Pair_Profit' else '.2%')
        )
        st.altair_chart((bar + text).properties(height=260), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_traffic:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### 👥 Store Traffic & Staffing Heatmap")
        if not traffic_df.empty:
            traffic_chart = alt.Chart(traffic_df).mark_bar(color='#FFC220', cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X('Day:N', title='', sort=cats, axis=alt.Axis(labelAngle=-45, labelColor='#555')),
                y=alt.Y('Transactions:Q', axis=None),
                tooltip=['Day', 'Transactions']
            ).properties(height=260)
            st.altair_chart(traffic_chart, use_container_width=True)
        else:
            st.info("No date data available to generate traffic heatmap.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 8. ACTION PLAN & TAKEAWAYS ---
    col_takeaways, col_action = st.columns([1, 1.8])
    
    with col_takeaways:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Executive Takeaways")
        takeaways = ""
        for i, row in rules_df.head(4).iterrows():
            action_text = row['Strategy'].split(':')[0].strip()
            takeaways += f"<li>Customers buying <strong>{row['Item_A']}</strong> heavily drive sales of <strong>{row['Item_B']}</strong>. Action: {action_text}</li>"
        st.markdown(f'<ul class="takeaways-list">{takeaways}</ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_action:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Intelligent Action Plan")
        display_cols = rules_df[['Item_A', 'Item_B', 'Strategy']].copy()
        display_cols.columns = ['Driver Product', 'Partner Product', 'Recommended Execution']
        st.dataframe(display_cols, hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 9. ANALYST SECTION ---
    with st.expander("🔍 Advanced Analytics (Money Left on the Table)"):
        st.markdown("""
        * **Missed Checkouts:** Estimated number of people who bought the Driver Product, but left without buying the Partner Product.
        * **Missed Profit:** The exact dollar amount lost by not bundling these items effectively.
        """)
        analyst_df = rules_df[['Item_A', 'Item_B', 'confidence', 'Missed_Txns', 'Missed_Profit']].copy()
        analyst_df['confidence'] = (analyst_df['confidence'] * 100).round(1).astype(str) + '%'
        analyst_df['Missed_Txns'] = analyst_df['Missed_Txns'].round(0).astype(int)
        analyst_df['Missed_Profit'] = analyst_df['Missed_Profit'].apply(lambda x: f"${x:,.2f}")
        analyst_df.columns = ['Driver (A)', 'Partner (B)', 'Checkout Probability', 'Missed Checkouts', 'Missed Profit ($)']
        st.dataframe(analyst_df, hide_index=True, use_container_width=True)
else:
    st.info("ℹ️ No strong product associations found based on your current filters and sensitivity level.")
