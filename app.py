import streamlit as st
import pandas as pd
import altair as alt
import io
import datetime
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Supercenter Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-banner {
        background-color: #0071CE;
        padding: 30px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: white;
    }
    .main-banner h1 {
        color: white; margin: 0; font-family: 'Arial', sans-serif; font-size: 34px; display: flex; align-items: center;
    }
    .kpi-container { display: flex; justify-content: space-between; gap: 15px; margin-bottom: 25px; }
    .kpi-card { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; flex: 1; text-align: center; }
    .kpi-title { color: #666666; font-size: 14px; font-weight: 600; text-transform: uppercase; margin-bottom: 10px; }
    .kpi-value { color: #0071CE; font-size: 34px; font-weight: bold; }
    
    .alert-box { background-color: #FFF3CD; border-left: 5px solid #FFC107; padding: 15px; border-radius: 4px; margin-bottom: 10px; color: #856404; font-weight: 500;}
    .critical-box { background-color: #F8D7DA; border-left: 5px solid #DC3545; padding: 15px; border-radius: 4px; margin-bottom: 30px; color: #721C24; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

# --- 2. INTERNAL DATABASE (Profit & Inventory) ---
# Embedded to keep the manager's experience to a single file upload
PRODUCT_DB = {
    'Bread': {'profit': 0.75, 'stock': 12}, # Low stock to trigger alert
    'Butter': {'profit': 1.10, 'stock': 85},
    'Milk': {'profit': 0.50, 'stock': 150},
    'Diapers': {'profit': 6.00, 'stock': 200},
    'Beer': {'profit': 4.50, 'stock': 300},
    'Wine': {'profit': 12.00, 'stock': 40},
    'Cheese': {'profit': 4.00, 'stock': 90},
    'Eggs': {'profit': 0.80, 'stock': 60},
    'Coffee': {'profit': 3.50, 'stock': 110},
    'Nutella': {'profit': 2.50, 'stock': 5} # Critical stock
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
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Check if weekend transactions are disproportionately high
    weekend_txns = df[df['Date'].dt.weekday >= 5].shape[0]
    weekday_txns = df[df['Date'].dt.weekday < 5].shape[0]
    
    # Normalize by days (2 weekend days vs 5 weekdays)
    if (weekend_txns / 2) > (weekday_txns / 5) * 1.5:
        return "📈 ANOMALY DETECTED: Weekend checkout volume is spiking over 50% above weekday averages. Ensure front-end registers are fully staffed on Saturdays."
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
    
    # Calculate Profitability for the Pair
    rules['Pair_Profit'] = rules.apply(
        lambda row: PRODUCT_DB.get(row['Item_A'], {}).get('profit', 0) + PRODUCT_DB.get(row['Item_B'], {}).get('profit', 0), 
        axis=1
    )
    
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Item_A'], row['Item_B']]), axis=1)
    rules = rules.sort_values(by='confidence', ascending=False).drop_duplicates(subset=['Pair_ID'], keep='first')
    
    # 🔄 DYNAMIC SORTING BASED ON MANAGER TOGGLE
    if optimize_for == "💰 Maximum Profit":
        rules = rules.sort_values(by='Pair_Profit', ascending=False)
    else:
        rules = rules.sort_values(by='support', ascending=False)
        
    rules['Strategy'] = rules.apply(lambda row: generate_strategy(row['Item_A'], row['Item_B'], row['lift'], row['confidence'], row['support']), axis=1)
    return rules

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("### 📥 Monthly Data Upload")
    uploaded_file = st.file_uploader("Upload transactions_2000.csv", type="csv")
    
    st.divider()
    st.markdown("### 🎯 Strategy Focus")
    optimization = st.radio(
        "Generate floor layout based on:",
        ["📦 Sales Volume", "💰 Maximum Profit"],
        help="Volume focuses on the most frequent purchases. Profit focuses on the highest margin pairings."
    )
    
    st.divider()
    sensitivity = st.slider("Rule Sensitivity", 0.01, 0.50, 0.05, 0.01)

# --- 5. DATA PROCESSING ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("👈 Please upload the generated 'transactions_2000.csv' in the sidebar.")
    st.stop()

total_txns = len(df)
all_items = [item.strip() for sublist in df['Items'].astype(str).str.split(',') for item in sublist]
avg_basket = round(len(all_items) / total_txns, 1) if total_txns > 0 else 0
best_seller = pd.Series(all_items).mode()[0] if all_items else "N/A"
total_revenue = sum([PRODUCT_DB.get(i, {}).get('profit', 1.00) for i in all_items])

# --- 6. MAIN DASHBOARD UI ---
dynamic_month = datetime.datetime.now().strftime("%B %Y")

st.markdown(f"""
<div class="main-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <h1>
                <svg width="45" height="45" viewBox="0 0 24 24" fill="none" style="margin-right: 12px;">
                  <path d="M12 0l1.4 7.6 6.8-3.4-3.4 6.8 7.6 1.4-7.6 1.4 3.4 6.8-6.8-3.4-1.4 7.6-1.4-7.6-6.8 3.4 3.4-6.8-7.6-1.4 7.6-1.4-3.4-6.8 6.8 3.4L12 0z" fill="#FFC220"/>
                </svg>
                Strategy Dashboard
            </h1>
        </div>
        <div style="background-color: #FFC220; color: #004c8c; padding: 6px 20px; border-radius: 50px; font-weight: bold; font-size: 16px;">
            {dynamic_month}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# 🚨 AI ALERT SECTION
anomaly = detect_anomalies(df)
if anomaly:
    st.markdown(f'<div class="alert-box">{anomaly}</div>', unsafe_allow_html=True)

rules_df = process_data(df, sensitivity, optimization)

# Inventory Alerts based on top strategies
if not rules_df.empty:
    for i, row in rules_df.head(5).iterrows():
        stock_b = PRODUCT_DB.get(row['Item_B'], {}).get('stock', 999)
        if stock_b < 20:
            st.markdown(f'<div class="critical-box">⚠️ INVENTORY CRITICAL: You only have {stock_b} units of <strong>{row["Item_B"]}</strong> remaining. Restock immediately to capture guaranteed follow-on sales from {row["Item_A"]}.</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card"><div class="kpi-title">🧾 Transactions</div><div class="kpi-value">{total_txns:,}</div></div>
    <div class="kpi-card"><div class="kpi-title">🛍️ Avg. Basket Size</div><div class="kpi-value">{avg_basket} items</div></div>
    <div class="kpi-card"><div class="kpi-title">⭐ Best-Selling Product</div><div class="kpi-value">{best_seller}</div></div>
    <div class="kpi-card"><div class="kpi-title">💰 Est. Total Profit</div><div class="kpi-value">${total_revenue:,.2f}</div></div>
</div>
""", unsafe_allow_html=True)

if not rules_df.empty:
    st.markdown(f"### 📊 Top 5 Product Pairings (Ranked by {optimization.split(' ')[1]})")
    
    chart_data = rules_df.head(5).copy()
    chart_data['Pair Name'] = chart_data['Item_A'] + " & " + chart_data['Item_B']
    
    if optimization == "💰 Maximum Profit":
        x_col = 'Pair_Profit'
        chart_title = 'Combined Profit per Checkout ($)'
    else:
        chart_data['% of Checkouts'] = chart_data['support'] * 100
        x_col = '% of Checkouts'
        chart_title = 'Percentage of Total Checkouts'
    
    base = alt.Chart(chart_data).encode(
        x=alt.X(f'{x_col}:Q', axis=None),
        y=alt.Y('Pair Name:N', sort='-x', title='', axis=alt.Axis(labelFontSize=14, tickSize=0, domain=False))
    )
    bar = base.mark_bar(color='#0071CE', cornerRadiusEnd=4, height=35)
    text = base.mark_text(align='left', baseline='middle', dx=5, fontSize=14, fontWeight='bold', color='#0071CE').encode(
        text=alt.Text(f'{x_col}:Q', format='$.2f' if x_col == 'Pair_Profit' else '.1f')
    )
    st.altair_chart((bar + text).properties(height=280), use_container_width=True)

    st.markdown("### 📋 Action Plan")
    display_cols = rules_df[['Item_A', 'Item_B', 'Strategy']].copy()
    display_cols.columns = ['Driver Product', 'Partner Product', 'Mathematically Backed Strategy']
    st.dataframe(display_cols, hide_index=True, use_container_width=True)
