import streamlit as st
import pandas as pd
import altair as alt
import os
import datetime
import math
import random
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. SETTINGS & TIMES NEW ROMAN STYLING ---
st.set_page_config(page_title="Store Manager Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Full screen and hide sidebar */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 98% !important; }
    [data-testid="collapsedControl"] { display: none !important; } 
    header { visibility: hidden !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    
    /* Simple, Modern Structure with Times New Roman Font */
    body { font-family: 'Times New Roman', Times, serif; background-color: #f4f6f9; color: #222;}
    
    /* Login Box */
    .login-container { max-width: 420px; margin: 10vh auto; padding: 40px; background: white; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.05); border: 1px solid #e0e0e0; text-align: center; font-family: 'Times New Roman', Times, serif;}
    
    /* Top Banner */
    .main-banner {
        background-color: #0071CE; padding: 20px 30px; border-radius: 8px; margin-bottom: 20px; 
        color: white; box-shadow: 0 4px 10px rgba(0, 113, 206, 0.15);
        display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
    }
    .main-banner h1 { margin: 0; font-size: 30px; font-weight: normal; display: flex; align-items: center; gap: 15px; font-family: 'Times New Roman', Times, serif;}
    .main-banner p { color: #FFC220; margin: 5px 0 0 0; font-size: 16px; }
    .date-badge { background-color: #FFC220; color: #004c8c; padding: 6px 20px; border-radius: 30px; font-weight: bold; font-size: 15px; }
    
    /* Daily Tasks (Briefing) */
    .briefing-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px; }
    .briefing-card { padding: 20px; border-radius: 8px; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.08); display: flex; flex-direction: column;}
    .briefing-card h4 { margin: 0 0 10px 0; font-size: 18px; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 8px;}
    .briefing-card p { margin: 0; font-size: 16px; line-height: 1.4;}
    .briefing-card ol { margin: 12px 0; padding-left: 20px; font-size: 15px; background: rgba(255,255,255,0.15); padding: 12px 12px 12px 28px; border-radius: 4px; }
    .impact-metric { margin-top: auto; font-size: 14px; font-weight: bold; background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px; text-align: center;}
    
    /* Quick Numbers (KPIs) */
    .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 25px; }
    .kpi-card { background-color: #ffffff; border-radius: 8px; padding: 20px 15px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.03); border: 1px solid #e0e0e0; }
    .kpi-title { color: #555; font-size: 14px; text-transform: uppercase; margin-bottom: 8px; font-weight: bold;}
    .kpi-value { color: #0071CE; font-size: 32px; font-weight: bold; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid #ccc;}
    .stTabs [data-baseweb="tab"] { height: 50px; font-size: 18px; font-family: 'Times New Roman', Times, serif; color: #333;}
</style>
""", unsafe_allow_html=True)

# Function to apply Times New Roman to Altair charts
def apply_chart_font(chart):
    return chart.configure_axis(
        labelFont='Times New Roman', titleFont='Times New Roman', labelFontSize=13, titleFontSize=14
    ).configure_legend(
        labelFont='Times New Roman', titleFont='Times New Roman', labelFontSize=13, titleFontSize=14
    ).configure_title(
        font='Times New Roman', fontSize=16
    ).configure_text(
        font='Times New Roman'
    )

walmart_spark_svg = """<svg width="40" height="40" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M50 10 L50 32 M85 30 L67 45 M85 70 L67 55 M50 90 L50 68 M15 70 L33 55 M15 30 L33 45" stroke="#FFC220" stroke-width="10" stroke-linecap="round"/></svg>"""

# --- 2. LOGIN SCREEN ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def check_password():
    if st.session_state['authenticated']: return True
    st.markdown(f"""
        <div class="login-container">
            <div style="margin-bottom: 10px;">{walmart_spark_svg}</div>
            <h2 style="color: #333; margin-bottom: 5px; font-family: 'Times New Roman', Times, serif;">Store Manager Login</h2>
            <p style="color: #666; font-size: 16px; margin-bottom: 25px;">Please enter your details to view store data.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("User Name")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log In", use_container_width=True):
                if username == "admin" and password == "manager123":
                    st.session_state['authenticated'] = True
                    st.rerun()
                else: st.error("Wrong user name or password.")
    return False

if not check_password(): st.stop()

# --- 3. STORE DATA (Products & Stock) ---
PRODUCT_DB = {
    'Bread': {'profit': 0.75, 'stock': 12}, 'Butter': {'profit': 1.10, 'stock': 85},
    'Milk': {'profit': 0.50, 'stock': 150}, 'Diapers': {'profit': 6.00, 'stock': 200},
    'Beer': {'profit': 4.50, 'stock': 300}, 'Wine': {'profit': 12.00, 'stock': 40},
    'Cheese': {'profit': 4.00, 'stock': 90}, 'Eggs': {'profit': 0.80, 'stock': 60},
    'Coffee': {'profit': 3.50, 'stock': 110}, 'Nutella': {'profit': 2.50, 'stock': 5},
    'Snacks': {'profit': 2.00, 'stock': 500}, 'Greeting Cards': {'profit': 3.00, 'stock': 400}
}

# --- 4. FINDING PATTERNS IN SALES ---
def find_sales_patterns(df, min_supp, total_txns):
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
        
    rules['Main Item'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Add-on Item'] = rules['consequents'].apply(lambda x: list(x)[0])
    rules['Profit Main'] = rules['Main Item'].apply(lambda x: PRODUCT_DB.get(x, {}).get('profit', 0))
    rules['Profit Add-on'] = rules['Add-on Item'].apply(lambda x: PRODUCT_DB.get(x, {}).get('profit', 0))
    
    rules['Missed Sales Count'] = ((rules['support'] / rules['confidence']) * total_txns) * (1 - rules['confidence'])
    rules['Money Left Behind'] = rules['Missed Sales Count'] * rules['Profit Add-on']
    
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Main Item'], row['Add-on Item']]), axis=1)
    return rules.sort_values(by='confidence', ascending=False).drop_duplicates(subset=['Pair_ID'], keep='first')

# --- 5. SETTINGS MENU ---
with st.expander("⚙️ System Settings (Upload Data & Log Out)"):
    admin_col1, admin_col2, admin_col3 = st.columns(3)
    with admin_col1: uploaded_file = st.file_uploader("Upload Store Receipts (CSV)", type="csv")
    with admin_col2: sensitivity = st.slider("Pattern Finder Sensitivity", 0.01, 0.50, 0.05, 0.01)
    with admin_col3: 
        st.write(""); st.write("")
        if st.button("Log Out of System", use_container_width=True):
            st.session_state['authenticated'] = False
            st.rerun()

# --- 6. READ DATA & ADD PAYMENT TYPES ---
# Generate sample data if no file is uploaded
if uploaded_file is not None: 
    df = pd.read_csv(uploaded_file)
else:
    # Create fake data that looks real for the manager
    items = list(PRODUCT_DB.keys())
    data = []
    start_date = datetime.datetime.now() - datetime.timedelta(days=7)
    for i in range(1, 1501):
        date = start_date + datetime.timedelta(days=random.randint(0, 7))
        basket = random.sample(items, random.randint(1, 4))
        # Add random payment type
        payment = random.choices(['Digital (Card/App)', 'Cash'], weights=[70, 30])[0]
        data.append([f"TXN-{8000+i}", date.strftime("%Y-%m-%d"), ",".join(basket), payment])
    df = pd.DataFrame(data, columns=['Transaction_ID', 'Date', 'Items', 'Payment_Type'])

df['Items_List'] = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])
df['Cart Size'] = df['Items_List'].apply(len)
df['Cart Profit'] = df['Items_List'].apply(lambda x: sum([PRODUCT_DB.get(i, {}).get('profit', 0) for i in x]))

total_txns = len(df)
all_items = [item for sublist in df['Items_List'] for item in sublist]
avg_basket = round(df['Cart Size'].mean(), 1) if total_txns > 0 else 0
avg_cart_profit = df['Cart Profit'].mean()
best_seller = pd.Series(all_items).mode()[0] if all_items else "None"
total_revenue = sum([PRODUCT_DB.get(i, {}).get('profit', 1.00) for i in all_items])

busiest_day, traffic_df = "N/A", pd.DataFrame()
if 'Date' in df.columns:
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        busiest_day = df['Date'].dt.day_name().mode()[0]
        traffic_df = df['Date'].dt.day_name().value_counts().reset_index()
        traffic_df.columns = ['Day', 'Customers']
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        traffic_df['Day'] = pd.Categorical(traffic_df['Day'], categories=cats, ordered=True)
        traffic_df = traffic_df.sort_values('Day')
        traffic_df['Registers Needed'] = (traffic_df['Customers'] / 40).apply(math.ceil)
    except: pass

rules_df = find_sales_patterns(df, sensitivity, total_txns)

# --- 7. DASHBOARD HEADER & TASKS ---
st.markdown(f"""
<div class="main-banner">
    <div>
        <h1>{walmart_spark_svg} Store Manager Dashboard</h1>
        <p>Simple tools to run a better, more profitable store.</p>
    </div>
    <div class="date-badge">Updated: {datetime.datetime.now().strftime("%B %d, %Y")}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📋 Today's Top 3 Tasks")
briefing_col1, briefing_col2, briefing_col3 = st.columns(3)

with briefing_col1:
    if not rules_df.empty:
        low_stock = rules_df.copy()
        low_stock['Stock'] = low_stock['Add-on Item'].apply(lambda x: PRODUCT_DB.get(x, {}).get('stock', 999))
        low_stock = low_stock[low_stock['Stock'] < 30].sort_values('Money Left Behind', ascending=False)
        if not low_stock.empty:
            oos_item = low_stock.iloc[0]['Add-on Item']
            st.markdown(f"""
            <div class="briefing-card" style="background-color: #DC3545;">
                <h4>1. Refill Empty Shelves</h4>
                <p><strong>{oos_item}</strong> is almost sold out.</p>
                <ol>
                    <li>Send a worker to the back room.</li>
                    <li>Bring all {oos_item} to the main aisle.</li>
                </ol>
                <div class="impact-metric">Prevents lost sales today</div>
            </div>
            """, unsafe_allow_html=True)
        else: 
            st.markdown(f'<div class="briefing-card" style="background-color: #28A745;"><h4>1. Stock Levels Good</h4><p>Your best-selling items are fully stocked.</p></div>', unsafe_allow_html=True)

with briefing_col2:
    if not rules_df.empty:
        top_bundle = rules_df.sort_values('Money Left Behind', ascending=False).iloc[0]
        st.markdown(f"""
        <div class="briefing-card" style="background-color: #0071CE;">
            <h4>2. Improve Shelf Layout</h4>
            <p>Put <strong>{top_bundle['Add-on Item']}</strong> next to <strong>{top_bundle['Main Item']}</strong>.</p>
            <ol>
                <li>Clear a display at the front of the store.</li>
                <li>Place both items side-by-side.</li>
            </ol>
            <div class="impact-metric">Can add ${top_bundle['Money Left Behind']:,.0f} in profit</div>
        </div>
        """, unsafe_allow_html=True)

with briefing_col3:
    if not traffic_df.empty:
        peak_day_row = traffic_df.loc[traffic_df['Customers'].idxmax()]
        st.markdown(f"""
        <div class="briefing-card" style="background-color: #FFC220; color: #222;">
            <h4 style="color: #222;">3. Plan Cashier Shifts</h4>
            <p>Your busiest day will be <strong>{peak_day_row['Day']}</strong>.</p>
            <ol>
                <li>Check the schedule for {peak_day_row['Day']}.</li>
                <li>Make sure you have <strong>{peak_day_row['Registers Needed']} cashiers</strong> working.</li>
            </ol>
            <div class="impact-metric" style="color: #222;">Keeps checkout lines short</div>
        </div>
        """, unsafe_allow_html=True)

st.download_button("📥 Print Today's Tasks for Employees", data="Task,Instructions\nRefill Shelves,Check backroom for low items.\nUpdate Layout,Move related items next to each other.\nManage Registers,Ensure enough cashiers during busy times.", file_name="Daily_Tasks.csv", mime="text/csv")

# Quick Numbers
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">Total Customers</div><div class="kpi-value">{total_txns:,}</div></div>
    <div class="kpi-card"><div class="kpi-title">Avg Items Per Cart</div><div class="kpi-value">{avg_basket}</div></div>
    <div class="kpi-card"><div class="kpi-title">Top Selling Item</div><div class="kpi-value">{best_seller}</div></div>
    <div class="kpi-card"><div class="kpi-title">Total Profit Made</div><div class="kpi-value">${total_revenue:,.0f}</div></div>
</div>
""", unsafe_allow_html=True)

# --- 8. STORE GROWTH TOOLS ---
st.markdown("### 📊 Store Growth Tools")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🛒 Total Cart Boosters", 
    "🏷️ Traffic Builders", 
    "💸 Missed Sales", 
    "👥 Staffing Needs",
    "💳 How People Pay",
    "🧹 Clearance Rack Finder"
])
    
with tab1:
    st.markdown("**Total Cart Boosters:** When a customer buys one of these items, they usually end up spending a lot more money in the store overall. Put these items in your weekly flyers.")
    halo_data = []
    for item in PRODUCT_DB.keys():
        contains_item = df[df['Items_List'].apply(lambda x: item in x)]
        if not contains_item.empty:
            halo_data.append({'Product': item, 'Avg_Cart_Profit': contains_item['Cart Profit'].mean()})
            
    halo_df = pd.DataFrame(halo_data).sort_values('Avg_Cart_Profit', ascending=False)
    if not halo_df.empty:
        bar_chart = alt.Chart(halo_df).mark_bar(color='#0071CE', cornerRadiusEnd=4, size=25).encode(
            x=alt.X('Avg_Cart_Profit:Q', title='Average Profit of Entire Cart ($)', axis=alt.Axis(format='$,.2f')),
            y=alt.Y('Product:N', sort='-x', title=''),
            tooltip=['Product', 'Avg_Cart_Profit']
        )
        rule = alt.Chart(pd.DataFrame({'baseline': [avg_cart_profit]})).mark_rule(color='#DC3545', strokeWidth=2, strokeDash=[5, 5]).encode(x='baseline:Q')
        st.altair_chart(apply_chart_font(bar_chart + rule).properties(height=350), use_container_width=True)

with tab2:
    st.markdown("**Traffic Builders (Loss Leaders):** These are cheap items (Main Item) that get people into the store to buy expensive items (Add-on Item). *Tip: Put the cheap items on sale.*")
    if not rules_df.empty:
        ll_df = rules_df[(rules_df['Profit Main'] < 1.50) & (rules_df['Profit Add-on'] > 3.00)].copy()
        if not ll_df.empty:
            ll_chart = alt.Chart(ll_df).mark_circle(size=400, opacity=0.9).encode(
                x=alt.X('confidence:Q', title='Chance of Buying the Expensive Item', axis=alt.Axis(format='%')),
                y=alt.Y('Profit Add-on:Q', title='Profit Made from Expensive Item ($)', axis=alt.Axis(format='$,.2f')),
                color=alt.Color('Main Item:N', title='Cheap Item (Put on sale)'),
                tooltip=['Main Item', 'Add-on Item', 'confidence']
            )
            st.altair_chart(apply_chart_font(ll_chart).properties(height=350), use_container_width=True)
        else: st.info("No patterns found. Adjust settings.")

with tab3:
    st.markdown("**Missed Sales (Bad Layout):** This shows how much money you are losing because related items are too far apart in the store.")
    if not rules_df.empty:
        rules_df['Item Pair'] = rules_df['Main Item'] + " + " + rules_df['Add-on Item']
        margin_chart = alt.Chart(rules_df.sort_values('Money Left Behind', ascending=False).head(6)).mark_bar(color='#28A745', cornerRadiusEnd=4, height=30).encode(
            x=alt.X('Money Left Behind:Q', title='Lost Profit ($)', axis=alt.Axis(format='$,.0f')),
            y=alt.Y('Item Pair:N', sort='-x', title=''),
            tooltip=['Item Pair', 'Money Left Behind']
        )
        st.altair_chart(apply_chart_font(margin_chart).properties(height=350), use_container_width=True)

with tab4:
    st.markdown("**Staffing Needs:** Shows how many customers visit each day and how many cashiers you need to schedule.")
    if not traffic_df.empty:
        base = alt.Chart(traffic_df).encode(x=alt.X('Day:N', sort=cats, title=''))
        bar = base.mark_bar(color='#E9ECEF', cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(y=alt.Y('Customers:Q', title='Number of Customers'))
        line = base.mark_line(color='#0071CE', strokeWidth=4).encode(y=alt.Y('Registers Needed:Q', title='Cashiers Needed'))
        points = base.mark_circle(color='#FFC220', size=150).encode(y=alt.Y('Registers Needed:Q'), tooltip=['Day', 'Customers', 'Registers Needed'])
        st.altair_chart(apply_chart_font(alt.layer(bar, line + points).resolve_scale(y='independent')).properties(height=350), use_container_width=True)

with tab5:
    st.markdown("**How Customers Pay:** See if your store relies more on cash or digital payments (cards/phones).")
    if 'Payment_Type' in df.columns:
        pay_df = df['Payment_Type'].value_counts().reset_index()
        pay_df.columns = ['Payment Type', 'Count']
        donut = alt.Chart(pay_df).mark_arc(innerRadius=80, stroke='#fff', strokeWidth=2).encode(
            theta='Count:Q',
            color=alt.Color('Payment Type:N', scale=alt.Scale(range=['#0071CE', '#FFC220']), legend=alt.Legend(title="", orient="right")),
            tooltip=['Payment Type', 'Count']
        )
        st.altair_chart(apply_chart_font(donut).properties(height=350), use_container_width=True)

with tab6:
    st.markdown("**Clearance Rack Finder:** These items are taking up space. You have a lot in the backroom, but customers rarely buy them. *Tip: Move these to a clearance bin.*")
    item_counts = pd.Series(all_items).value_counts().reset_index()
    item_counts.columns = ['Item', 'Times Bought']
    item_counts['Current Stock'] = item_counts['Item'].apply(lambda x: PRODUCT_DB.get(x, {}).get('stock', 0))
    
    # Define dead stock: High inventory (>100), low sales (Bought in less than 5% of trips)
    dead_stock = item_counts[(item_counts['Current Stock'] > 100) & (item_counts['Times Bought'] < (total_txns * 0.05))]
    
    if not dead_stock.empty:
        st.dataframe(dead_stock[['Item', 'Current Stock', 'Times Bought']], hide_index=True, use_container_width=True)
    else:
        st.success("Great news! You don't have any major dead stock taking up space right now.")
