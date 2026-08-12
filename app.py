import streamlit as st
import pandas as pd
import altair as alt
import os
import datetime
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Supercenter Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Global Typography & Spacing */
    body { font-family: 'Inter', 'Segoe UI', sans-serif; background-color: #f4f6f9; }
    
    /* Main Banner */
    .main-banner {
        background-color: #0071CE; padding: 25px 35px; border-radius: 12px; margin-bottom: 30px; 
        color: white; box-shadow: 0 4px 20px rgba(0, 113, 206, 0.15);
        display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
    }
    .main-banner h1 { margin: 0; font-size: 30px; display: flex; align-items: center; gap: 15px; font-weight: 700; }
    .main-banner p { color: #FFC220; margin: 5px 0 0 0; font-size: 16px; font-weight: 500; opacity: 0.95; }
    .date-badge { background-color: #FFC220; color: #004c8c; padding: 8px 24px; border-radius: 50px; font-weight: 800; font-size: 15px; letter-spacing: 0.5px; }
    
    /* Floating KPI Cards */
    .kpi-container { display: flex; justify-content: space-between; gap: 20px; margin-bottom: 30px; }
    .kpi-card { 
        background-color: #ffffff; border-radius: 12px; padding: 25px 20px; flex: 1; text-align: center; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.03); transition: transform 0.2s ease; border: 1px solid #f0f0f0;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.06); }
    .kpi-title { color: #888888; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .kpi-value { color: #0071CE; font-size: 32px; font-weight: 800; }
    
    /* Alerts */
    .alert-box { background-color: #fff9e6; border-left: 6px solid #FFC107; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; color: #856404; font-weight: 500; box-shadow: 0 2px 8px rgba(0,0,0,0.02);}
    .critical-box { background-color: #fff0f1; border-left: 6px solid #DC3545; padding: 16px 20px; border-radius: 8px; margin-bottom: 20px; color: #721C24; font-weight: 500; box-shadow: 0 2px 8px rgba(0,0,0,0.02);}
    
    /* Layout Cards */
    .content-card { background-color: #ffffff; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #f0f0f0; margin-bottom: 30px; height: 100%;}
    .takeaways-list { list-style: none; padding-left: 0; margin: 0; }
    .takeaways-list li { margin-bottom: 16px; padding-left: 24px; position: relative; font-size: 15px; color: #444; line-height: 1.5; }
    .takeaways-list li:before { content: '→'; position: absolute; left: 0; color: #0071CE; font-weight: bold; }
    
    /* Hide Streamlit elements to make it feel like an app */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 2. INTERNAL DATABASE (Profit & Inventory) ---
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

# --- 3. HELPER FUNCTIONS ---
def generate_strategy(item_a, item_b, lift, conf, supp):
    conf_pct = round(conf * 100)
    if conf >= 0.95:
        return f"🏷️ DISCOUNT {item_a}: Drive full-price sales of {item_b} (⭐ Star Product). 100% of {item_a} buyers also buy {item_b}."
    elif lift > 1.5 and conf >= 0.6:
        return f"📦 BUNDLE: Place {item_a} & {item_b} on the same endcap. {conf_pct}% chance they are bought together."
    else:
        return f"🚶 SEPARATE: Place at opposite ends. Appears in {round(supp*100)}% of checkouts; forces foot traffic."

def detect_anomalies(df):
    if 'Date' not in df.columns: return None
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        weekend_txns = df[df['Date'].dt.weekday >= 5].shape[0]
        weekday_txns = df[df['Date'].dt.weekday < 5].shape[0]
        if (weekend_txns / 2) > (weekday_txns / 5) * 1.5:
            return "📈 ANOMALY DETECTED: Weekend checkout volume is spiking over 50% above weekday averages. Ensure front-end registers are fully staffed on Saturdays."
    except:
        pass
    return None

def process_data(df, min_supp, optimize_for):
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
    
    rules['Pair_Profit'] = rules.apply(
        lambda row: PRODUCT_DB.get(row['Item_A'], {}).get('profit', 0) + PRODUCT_DB.get(row['Item_B'], {}).get('profit', 0), 
        axis=1
    )
    
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Item_A'], row['Item_B']]), axis=1)
    rules = rules.sort_values(by='confidence', ascending=False).drop_duplicates(subset=['Pair_ID'], keep='first')
    
    if optimize_for == "💰 Maximum Profit":
        rules = rules.sort_values(by='Pair_Profit', ascending=False)
    else:
        rules = rules.sort_values(by='support', ascending=False)
        
    rules['Strategy'] = rules.apply(lambda row: generate_strategy(row['Item_A'], row['Item_B'], row['lift'], row['confidence'], row['support']), axis=1)
    return rules

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Walmart_logo.svg/512px-Walmart_logo.svg.png", width=120)
    st.markdown("### 📥 Data Manager")
    st.caption("The dashboard is synced to your repository. Upload a file to manually override.")
    uploaded_file = st.file_uploader("Override Transactions", type="csv", label_visibility="collapsed")
    
    st.divider()
    
    st.markdown("### 🎯 Strategy Focus")
    optimization = st.radio(
        "Optimize floor layout for:",
        ["📦 Sales Volume", "💰 Maximum Profit"],
        help="Volume prioritizes frequently bought items. Profit prioritizes highest combined margin pairings.",
        label_visibility="collapsed"
    )
    
    st.divider()
    
    st.markdown("### ⚙️ Engine Sensitivity")
    sensitivity = st.slider("Rule Sensitivity", 0.01, 0.50, 0.05, 0.01, label_visibility="collapsed")
    st.caption("Lowering this threshold finds rarer patterns. Raising it finds only the most common.")

# --- 5. DATA INGESTION ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists("transactions_2000.csv"):
    df = pd.read_csv("transactions_2000.csv")
else:
    st.error("⚠️ 'transactions_2000.csv' not found. Please upload a file via the sidebar to continue.")
    st.stop()

total_txns = len(df)
all_items = [item.strip() for sublist in df['Items'].astype(str).str.split(',') for item in sublist]
avg_basket = round(len(all_items) / total_txns, 1) if total_txns > 0 else 0
best_seller = pd.Series(all_items).mode()[0] if all_items else "N/A"
total_revenue = sum([PRODUCT_DB.get(i, {}).get('profit', 1.00) for i in all_items])

# --- 6. MAIN DASHBOARD UI ---
dynamic_month = datetime.datetime.now().strftime("%B %Y")

# Header Banner
st.markdown(f"""
<div class="main-banner">
    <div>
        <h1>Strategy Dashboard</h1>
        <p>Data-driven layout and pricing intelligence</p>
    </div>
    <div class="date-badge">
        {dynamic_month}
    </div>
</div>
""", unsafe_allow_html=True)

# Alerts Area
anomaly = detect_anomalies(df)
if anomaly:
    st.markdown(f'<div class="alert-box">{anomaly}</div>', unsafe_allow_html=True)

rules_df = process_data(df, sensitivity, optimization)

if not rules_df.empty:
    for i, row in rules_df.head(5).iterrows():
        stock_b = PRODUCT_DB.get(row['Item_B'], {}).get('stock', 999)
        if stock_b < 20:
            st.markdown(f'<div class="critical-box">⚠️ <strong>INVENTORY CRITICAL:</strong> Only {stock_b} units of {row["Item_B"]} remaining. Restock immediately to capture guaranteed follow-on sales from {row["Item_A"]}.</div>', unsafe_allow_html=True)

# KPI Row
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card"><div class="kpi-title">🧾 Transactions Analyzed</div><div class="kpi-value">{total_txns:,}</div></div>
    <div class="kpi-card"><div class="kpi-title">🛍️ Avg. Basket Size</div><div class="kpi-value">{avg_basket} <span style="font-size:16px; font-weight:600; color:#888;">items</span></div></div>
    <div class="kpi-card"><div class="kpi-title">⭐ Best-Selling Product</div><div class="kpi-value">{best_seller}</div></div>
    <div class="kpi-card"><div class="kpi-title">💰 Est. Total Profit</div><div class="kpi-value">${total_revenue:,.2f}</div></div>
</div>
""", unsafe_allow_html=True)

# Dashboard Core: 2-Column Layout
if not rules_df.empty:
    col1, col2 = st.columns([1.8, 1.2]) # 60% / 40% split
    
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(f"### 📊 Top 5 Product Pairings <span style='font-size: 14px; color: #888; font-weight: normal;'>(Ranked by {optimization.split(' ')[1]})</span>", unsafe_allow_html=True)
        
        chart_data = rules_df.head(5).copy()
        chart_data['Pair Name'] = chart_data['Item_A'] + " & " + chart_data['Item_B']
        
        if optimization == "💰 Maximum Profit":
            x_col = 'Pair_Profit'
        else:
            chart_data['% of Checkouts'] = chart_data['support'] * 100
            x_col = '% of Checkouts'
        
        base = alt.Chart(chart_data).encode(
            x=alt.X(f'{x_col}:Q', axis=None),
            y=alt.Y('Pair Name:N', sort='-x', title='', axis=alt.Axis(labelFontSize=13, labelColor='#555', tickSize=0, domain=False))
        )
        bar = base.mark_bar(color='#0071CE', cornerRadiusEnd=4, height=32)
        text = base.mark_text(align='left', baseline='middle', dx=8, fontSize=13, fontWeight='bold', color='#0071CE').encode(
            text=alt.Text(f'{x_col}:Q', format='$.2f' if x_col == 'Pair_Profit' else '.1f')
        )
        st.altair_chart((bar + text).properties(height=260), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Executive Takeaways")
        takeaways = ""
        for i, row in rules_df.head(4).iterrows():
            action_text = row['Strategy'].split(':')[0].strip()
            takeaways += f"<li>Customers buying <strong>{row['Item_A']}</strong> heavily drive sales of <strong>{row['Item_B']}</strong>. Action: {action_text}</li>"
        
        st.markdown(f'<ul class="takeaways-list">{takeaways}</ul>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Action Plan Full Width
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Complete Action Plan")
    display_cols = rules_df[['Item_A', 'Item_B', 'Strategy']].copy()
    display_cols.columns = ['Driver Product', 'Partner Product', 'Recommended Execution Strategy']
    st.dataframe(display_cols, hide_index=True, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Analyst Expander (Clean styling)
    with st.expander("🔍 Advanced Analytics (For Data Teams)"):
        st.markdown("""
        * **Support %:** Baseline frequency. How often this specific pair of items appears across *all* checkouts.
        * **Confidence %:** Conditional probability. When Product A is purchased, the likelihood Product B is also purchased.
        * **Lift:** Correlation strength. A Lift > 1.0 indicates a positive, intentional pairing beyond random chance.
        """)
        
        analyst_df = rules_df[['Item_A', 'Item_B', 'support', 'confidence', 'lift']].copy()
        analyst_df['support'] = (analyst_df['support'] * 100).round(1).astype(str) + '%'
        analyst_df['confidence'] = (analyst_df['confidence'] * 100).round(1).astype(str) + '%'
        analyst_df['lift'] = analyst_df['lift'].round(2)
        analyst_df.columns = ['Driver (A)', 'Partner (B)', 'Support %', 'Confidence %', 'Lift Factor']
        st.dataframe(analyst_df, hide_index=True, use_container_width=True)
else:
    st.info("ℹ️ No strong product associations found at current sensitivity level. Try lowering the threshold in the sidebar.")
