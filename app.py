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
    .main-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #E5E7EB; }
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
</style>
""", unsafe_allow_html=True)

# 100% BULLETPROOF INLINE SVG LOGO
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
PRODUCT_DB = {
    'Bread': {'profit': 0.75, 'stock': 12}, 'Butter': {'profit': 1.10, 'stock': 85},
    'Milk': {'profit': 0.50, 'stock': 150}, 'Diapers': {'profit': 6.00, 'stock': 200},
    'Beer': {'profit': 4.50, 'stock': 300}, 'Wine': {'profit': 12.00, 'stock': 40},
    'Cheese': {'profit': 4.00, 'stock': 90}, 'Eggs': {'profit': 0.80, 'stock': 60},
    'Coffee': {'profit': 3.50, 'stock': 110}, 'Nutella': {'profit': 2.50, 'stock': 5},
    'Snacks': {'profit': 2.00, 'stock': 500}
}

@st.cache_data(ttl=300)
def fetch_live_erp_data():
    items = list(PRODUCT_DB.keys())
    data = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=7)
    
    for i in range(1, 1501):
        date = start_date + datetime.timedelta(days=random.randint(0, 7))
        if date.weekday() >= 5: 
            basket = ['Beer', 'Diapers']
            if random.random() > 0.5: basket.append('Snacks')
        else:
            basket = random.sample(items, random.randint(1, 4))
            if 'Wine' in basket and 'Cheese' not in basket and random.random() > 0.2: basket.append('Cheese')
        data.append([f"TXN-{80000+i}", date.strftime("%Y-%m-%d"), ",".join(basket)])
        
    return pd.DataFrame(data, columns=['Transaction_ID', 'Date', 'Items'])

# --- 4. DATA PROCESSING ENGINE ---
def generate_strategy(item_a, item_b, lift, conf, supp, missed_profit):
    if conf >= 0.95: return f"🏷️ DISCOUNT: Drive sales of {item_b}."
    elif lift > 1.5 and conf >= 0.6: return f"📦 BUNDLE: Capture ${missed_profit:,.0f} lost profit."
    else: return f"🚶 SEPARATE: Force store navigation."

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
    
    rules['Strategy'] = rules.apply(lambda row: generate_strategy(row['Item_A'], row['Item_B'], row['lift'], row['confidence'], row['support'], row['Missed_Profit']), axis=1)
    
    # Safely deduplicate unordered pairs
    rules['Pair_ID'] = rules.apply(lambda row: " & ".join(sorted([row['Item_A'], row['Item_B']])), axis=1)
    rules = rules.sort_values(by='confidence', ascending=False).drop_duplicates(subset=['Pair_ID'], keep='first')
    
    if optimize_for == "💰 Maximum Profit": 
        return rules.sort_values(by='Pair_Profit', ascending=False)
    else: 
        return rules.sort_values(by='support', ascending=False)

# --- 5. TOP LEVEL NAVIGATION & SYNC ---
header_col1, header_col2 = st.columns([1, 1])
with header_col1:
    st.markdown(f'<div class="main-header"><div class="header-title">{walmart_spark_svg} AI Strategy Engine</div></div>', unsafe_allow_html=True)
with header_col2:
    st.write("") 
    sync_col1, sync_col2 = st.columns([3, 1])
    with sync_col1:
        st.markdown('<div style="text-align: right; margin-top: 5px;"><span class="live-badge"><span class="live-dot"></span>LIVE API CONNECTION ACTIVE</span></div>', unsafe_allow_html=True)
    with sync_col2:
        if st.button("🔄 Sync ERP", use_container_width=True):
            with st.spinner("Connecting to ERP..."):
                time.sleep(1.2)
                st.cache_data.clear()
                st.rerun()

# --- 6. ON-PAGE CONTROLS ---
st.markdown("<div style='margin-bottom: 20px; font-weight: 600; color: #111827;'>🎛️ Diagnostic Controls</div>", unsafe_allow_html=True)
ctrl_1, ctrl_2, ctrl_3 = st.columns([1.5, 1.5, 1])
with ctrl_1:
    optimization = st.radio("Optimization Target:", ["📦 Sales Volume", "💰 Maximum Profit"], horizontal=True, label_visibility="collapsed")
with ctrl_2:
    sensitivity = st.slider("Algorithm Sensitivity", 0.01, 0.50, 0.05, 0.01, label_visibility="collapsed")
with ctrl_3:
    if st.button("🔒 Secure Logout", use_container_width=True):
        st.session_state['authenticated'] = False
        st.rerun()

# Load Data
df = fetch_live_erp_data()
df['Items_List'] = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])

# Metrics
total_txns = len(df)
all_items = [item for sublist in df['Items_List'] for item in sublist]
total_revenue = sum([PRODUCT_DB.get(i, {}).get('profit', 1.00) for i in all_items])
health_score = min(100, int((total_revenue / 5000) * 100)) 

# --- 7. KPI GRID ---
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">Store Health Score</div><div class="kpi-value">{health_score}/100</div><div class="kpi-trend trend-up">↑ Optimal Range</div></div>
    <div class="kpi-card"><div class="kpi-title">7-Day Transaction Vol</div><div class="kpi-value">{total_txns:,}</div><div class="kpi-trend trend-up">↑ Live Data</div></div>
    <div class="kpi-card"><div class="kpi-title">Est. Gross Profit</div><div class="kpi-value">${total_revenue:,.0f}</div><div class="kpi-trend trend-up">↑ +4.2% vs last wk</div></div>
    <div class="kpi-card"><div class="kpi-title">Avg Basket Size</div><div class="kpi-value">{round(df['Items_List'].apply(len).mean(), 1)}</div><div class="kpi-trend trend-down">↓ -0.1 vs avg</div></div>
</div>
""", unsafe_allow_html=True)

# --- 8. CORE DIAGNOSTICS ---
rules_df = process_data(df, sensitivity, optimization, total_txns)

if not rules_df.empty:
    for i, row in rules_df.head(2).iterrows():
        stock = PRODUCT_DB.get(row['Item_B'], {}).get('stock', 999)
        if stock < 20:
            st.markdown(f'<div class="critical-box">⚠️ <strong>SYSTEM ALERT:</strong> {row["Item_B"]} inventory is critical ({stock} units). This will block algorithm-predicted sales of {row["Item_A"]}.</div>', unsafe_allow_html=True)

    col_chart, col_text = st.columns([1.5, 1])
    with col_chart:
        st.markdown(f'<div class="content-card"><div class="card-header">Highest Performing Pairings ({optimization.split(" ")[1]})</div>', unsafe_allow_html=True)
        chart_data = rules_df.head(5).copy()
        chart_data['Pair'] = chart_data['Item_A'] + " + " + chart_data['Item_B']
        x_col = 'Pair_Profit' if optimization == "💰 Maximum Profit" else 'support'
        
        base = alt.Chart(chart_data).encode(
            x=alt.X(f'{x_col}:Q', axis=None), y=alt.Y('Pair:N', sort='-x', title='', axis=alt.Axis(labelFontSize=13, tickSize=0, domain=False))
        )
        bar = base.mark_bar(color='#0071CE', cornerRadiusEnd=6, height=35)
        text = base.mark_text(align='left', dx=8, fontSize=13, fontWeight='600', color='#111827').encode(
            text=alt.Text(f'{x_col}:Q', format='$.2f' if x_col == 'Pair_Profit' else '.1%')
        )
        st.altair_chart((bar + text).properties(height=260), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_text:
        st.markdown('<div class="content-card"><div class="card-header">Automated Directives</div>', unsafe_allow_html=True)
        display_df = rules_df[['Item_A', 'Item_B', 'Strategy', 'confidence', 'Missed_Profit']].head(6).copy()
        display_df['Action'] = display_df['Strategy'].apply(lambda x: x.split(':')[0])
        display_df['Conf'] = (display_df['confidence'] * 100).round(0).astype(str) + '%'
        display_df['Value'] = display_df['Missed_Profit'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df[['Item_A', 'Item_B', 'Action', 'Conf', 'Value']], hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Insufficient data to generate confidence rules. Adjust parameters.")
