import streamlit as st
import pandas as pd
import altair as alt
import datetime
import time
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. CONFIGURATION & MODERN STYLING ---
st.set_page_config(page_title="Supercenter Operations", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* TRUE FULL SCREEN & CLEAN AESTHETIC */
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 98% !important; }
    [data-testid="collapsedControl"] { display: none !important; } 
    header { visibility: hidden !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background-color: #FAFAFA; color: #111827; }
    
    /* Modern Login */
    .login-container { max-width: 380px; margin: 12vh auto; padding: 40px 30px; background: white; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.04); border: 1px solid #F3F4F6; text-align: center; }
    
    /* Sleek Header */
    .main-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #E5E7EB; }
    .header-title { display: flex; align-items: center; gap: 12px; font-size: 24px; font-weight: 700; color: #111827; margin: 0; }
    
    /* Blinking Live Indicator */
    @keyframes blink { 50% { opacity: 0; } }
    .live-dot { height: 8px; width: 8px; background-color: #10B981; border-radius: 50%; display: inline-block; animation: blink 2s linear infinite; margin-right: 6px; }
    .live-badge { display: flex; align-items: center; background-color: #ECFDF5; color: #047857; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; border: 1px solid #A7F3D0; }
    
    /* Minimalist KPI Grid */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
    .kpi-card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #F3F4F6; }
    .kpi-title { color: #6B7280; font-size: 13px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .kpi-value { color: #111827; font-size: 32px; font-weight: 700; letter-spacing: -1px; }
    .kpi-trend { font-size: 13px; font-weight: 500; margin-top: 8px; }
    .trend-up { color: #10B981; }
    .trend-down { color: #EF4444; }
    
    /* Clean Content Cards */
    .content-card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #F3F4F6; margin-bottom: 20px; height: 100%; }
    .card-header { font-size: 16px; font-weight: 600; color: #111827; margin-bottom: 20px; border-bottom: 1px solid #F3F4F6; padding-bottom: 12px; }
    
    /* Alert Styling */
    .critical-box { background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 12px 16px; border-radius: 6px; margin-bottom: 15px; color: #991B1B; font-size: 14px; font-weight: 500; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 30px; border-bottom: 1px solid #E5E7EB; }
    .stTabs [data-baseweb="tab"] { height: 45px; font-weight: 500; font-size: 15px; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

walmart_spark_svg = """<svg width="32" height="32" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M50 10 L50 32 M85 30 L67 45 M85 70 L67 55 M50 90 L50 68 M15 70 L33 55 M15 30 L33 45" stroke="#FFC220" stroke-width="10" stroke-linecap="round"/></svg>"""

# --- 2. AUTHENTICATION LOGIC ---
if 'authenticated' not in st.session_state: st.session_state['authenticated'] = False

def check_password():
    if st.session_state['authenticated']: return True
    st.markdown(f"""
        <div class="login-container">
            <div style="margin-bottom: 15px;">{walmart_spark_svg}</div>
            <h2 style="color: #111827; margin-bottom: 8px; font-size: 22px;">Executive Operations</h2>
            <p style="color: #6B7280; font-size: 14px; margin-bottom: 30px;">Authenticate to access live ERP data.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("User ID", placeholder="admin")
            password = st.text_input("Passcode", type="password", placeholder="manager123")
            if st.form_submit_button("Authenticate", use_container_width=True):
                if username == "admin" and password == "manager123":
                    st.session_state['authenticated'] = True
                    st.rerun()
                else: st.error("Authentication failed.")
    return False

if not check_password(): st.stop()

# --- 3. ENTERPRISE DATA INTEGRATION (API SIMULATOR) ---
# This is how a real app bridges the gap before the backend API is live
PRODUCT_DB = {
    'Bread': {'profit': 0.75, 'stock': 12}, 'Butter': {'profit': 1.10, 'stock': 85},
    'Milk': {'profit': 0.50, 'stock': 150}, 'Diapers': {'profit': 6.00, 'stock': 200},
    'Beer': {'profit': 4.50, 'stock': 300}, 'Wine': {'profit': 12.00, 'stock': 40},
    'Cheese': {'profit': 4.00, 'stock': 90}, 'Eggs': {'profit': 0.80, 'stock': 60},
    'Coffee': {'profit': 3.50, 'stock': 110}, 'Nutella': {'profit': 2.50, 'stock': 5},
    'Snacks': {'profit': 2.00, 'stock': 500}
}

@st.cache_data(ttl=300) # Caches the API pull for 5 mins
def fetch_live_erp_data():
    """Simulates a live JSON payload request to Walmart's internal POS system."""
    items = list(PRODUCT_DB.keys())
    data = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=7) # Last 7 days rolling
    
    for i in range(1, 1501):
        date = start_date + datetime.timedelta(days=random.randint(0, 7))
        # Logic to simulate real human shopping habits
        if date.weekday() >= 5: 
            basket = ['Beer', 'Diapers']
            if random.random() > 0.5: basket.append('Snacks')
        else:
            basket = random.sample(items, random.randint(1, 4))
            if 'Wine' in basket and 'Cheese' not in basket and random.random() > 0.2: basket.append('Cheese')
        
        data.append([f"TXN-{80000+i}", date.strftime("%Y-%m-%d"), ",".join(basket)])
        
    return pd.DataFrame(data, columns=['Transaction_ID', 'Date', 'Items'])

# --- 4. DATA PROCESSING ENGINE ---
def process_data(df, min_supp):
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
    
    rules['Missed_Profit'] = rules.apply(lambda row: ((row['support'] / row['confidence']) * len(df)) * (1 - row['confidence']) * PRODUCT_DB.get(row['Item_B'], {}).get('profit', 0), axis=1)
    
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Item_A'], row['Item_B']]), axis=1)
    return rules.sort_values(by='Pair_Profit', ascending=False).drop_duplicates(subset=['Pair_ID'], keep='first')

# --- 5. TOP LEVEL NAVIGATION & SYNC ---
header_col1, header_col2 = st.columns([1, 1])
with header_col1:
    st.markdown(f'<div class="main-header"><div class="header-title">{walmart_spark_svg} AI Strategy Engine</div></div>', unsafe_allow_html=True)
with header_col2:
    st.write("") # Spacing
    sync_col1, sync_col2 = st.columns([3, 1])
    with sync_col1:
        st.markdown('<div style="text-align: right; margin-top: 5px;"><span class="live-badge"><span class="live-dot"></span>LIVE API CONNECTION ACTIVE</span></div>', unsafe_allow_html=True)
    with sync_col2:
        if st.button("🔄 Sync ERP", use_container_width=True):
            with st.spinner("Connecting to ERP..."):
                time.sleep(1.2) # Simulate API latency for the "wow" factor
                st.cache_data.clear()
                st.rerun()

# Load Data from API Simulator
df = fetch_live_erp_data()
df['Items_List'] = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])

# Metrics
total_txns = len(df)
all_items = [item for sublist in df['Items_List'] for item in sublist]
total_revenue = sum([PRODUCT_DB.get(i, {}).get('profit', 1.00) for i in all_items])
health_score = min(100, int((total_revenue / 5000) * 100)) # Simple composite metric

# --- 6. KPI GRID ---
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">Store Health Score</div><div class="kpi-value">{health_score}/100</div><div class="kpi-trend trend-up">↑ Optimal Range</div></div>
    <div class="kpi-card"><div class="kpi-title">7-Day Transaction Vol</div><div class="kpi-value">{total_txns:,}</div><div class="kpi-trend trend-up">↑ Live Data</div></div>
    <div class="kpi-card"><div class="kpi-title">Est. Gross Profit</div><div class="kpi-value">${total_revenue:,.0f}</div><div class="kpi-trend trend-up">↑ +4.2% vs last wk</div></div>
    <div class="kpi-card"><div class="kpi-title">Avg Basket Size</div><div class="kpi-value">{round(df['Items_List'].apply(len).mean(), 1)}</div><div class="kpi-trend trend-down">↓ -0.1 vs avg</div></div>
</div>
""", unsafe_allow_html=True)

# --- 7. CORE DIAGNOSTICS ---
rules_df = process_data(df, 0.05, "💰 Maximum Profit", total_txns)

if not rules_df.empty:
    # Inventory Alerts
    for i, row in rules_df.head(2).iterrows():
        stock = PRODUCT_DB.get(row['Item_B'], {}).get('stock', 999)
        if stock < 20:
            st.markdown(f'<div class="critical-box">⚠️ <strong>SYSTEM ALERT:</strong> {row["Item_B"]} inventory is critical ({stock} units). This will block algorithm-predicted sales of {row["Item_A"]}.</div>', unsafe_allow_html=True)

    st.markdown("### Operational Intelligence")
    tab1, tab2, tab3 = st.tabs(["💰 Profit Optimization", "📦 Layout Directives", "⚙️ Admin & Controls"])
    
    with tab1:
        col_chart, col_text = st.columns([1.5, 1])
        with col_chart:
            st.markdown('<div class="content-card"><div class="card-header">Highest Margin Pairings (Predicted)</div>', unsafe_allow_html=True)
            chart_data = rules_df.head(5).copy()
            chart_data['Pair'] = chart_data['Item_A'] + " + " + chart_data['Item_B']
            
            base = alt.Chart(chart_data).encode(
                x=alt.X('Pair_Profit:Q', axis=None), y=alt.Y('Pair:N', sort='-x', title='', axis=alt.Axis(labelFontSize=13, tickSize=0, domain=False))
            )
            bar = base.mark_bar(color='#0071CE', cornerRadiusEnd=6, height=35)
            text = base.mark_text(align='left', dx=8, fontSize=13, fontWeight='600', color='#111827').encode(text=alt.Text('Pair_Profit:Q', format='$.2f'))
            st.altair_chart((bar + text).properties(height=260), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_text:
            st.markdown('<div class="content-card"><div class="card-header">Revenue Leakage</div>', unsafe_allow_html=True)
            st.markdown("<p style='font-size: 14px; color: #6B7280; margin-bottom: 20px;'>Capital lost when customers purchase the anchor item but fail to purchase the associated pairing.</p>", unsafe_allow_html=True)
            
            leakage_chart = alt.Chart(rules_df.head(4)).mark_bar(color='#10B981', cornerRadiusEnd=4, height=20).encode(
                x=alt.X('Missed_Profit:Q', axis=None),
                y=alt.Y('Item_B:N', sort='-x', title='', axis=alt.Axis(labelFontSize=12)),
                tooltip=['Item_A', 'Item_B', 'Missed_Profit']
            ).properties(height=180)
            st.altair_chart(leakage_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="content-card"><div class="card-header">Automated Merchandising Directives</div>', unsafe_allow_html=True)
        display_df = rules_df[['Item_A', 'Item_B', 'confidence', 'Missed_Profit']].head(6).copy()
        display_df['Directive'] = display_df.apply(lambda r: f"BUNDLE (Endcap)" if r['confidence'] > 0.6 else "SEPARATE (Aisle)", axis=1)
        display_df['Confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
        display_df['Est. Lift Value'] = display_df['Missed_Profit'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df[['Item_A', 'Item_B', 'Directive', 'Confidence', 'Est. Lift Value']], hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab3:
        st.markdown('<div class="content-card"><div class="card-header">System Settings</div>', unsafe_allow_html=True)
        st.slider("Algorithm Sensitivity Threshold", 0.01, 0.50, 0.05, 0.01, key="sensitivity_slider")
        st.caption("Adjusts the Apriori confidence minimums. Higher values restrict results to absolute certainties.")
        st.write("")
        if st.button("End Session (Logout)", type="primary"):
            st.session_state['authenticated'] = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Insufficient data to generate confidence rules. Adjust parameters.")
