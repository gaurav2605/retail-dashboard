import streamlit as st
import pandas as pd
import altair as alt
import io
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Supercenter Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Walmart-inspired minimalist design
st.markdown("""
<style>
    /* Main Banner */
    .main-banner {
        background-color: #0071CE;
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 30px;
        color: white;
    }
    .main-banner h1 {
        color: white;
        margin: 0;
        font-family: 'Arial', sans-serif;
        font-size: 36px;
    }
    .main-banner p {
        color: #FFC220;
        margin: 5px 0 0 0;
        font-size: 20px;
        font-weight: 600;
    }
    
    /* KPI Cards */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 40px;
    }
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        flex: 1;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .kpi-title {
        color: #666666;
        font-size: 15px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #0071CE;
        font-size: 32px;
        font-weight: bold;
    }
    
    /* Takeaways Box */
    .takeaways-box {
        background-color: #f4f8fb;
        border-left: 5px solid #FFC220;
        padding: 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 40px;
    }
    .takeaways-box h3 {
        margin-top: 0;
        color: #333;
        font-size: 18px;
    }
    .takeaways-box ul {
        margin-bottom: 0;
        font-size: 16px;
        color: #444;
    }
    .takeaways-box li {
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- BUILT-IN SAMPLE DATA ---
SAMPLE_CSV = """Transaction_ID,Items
T101,"Milk,Bread,Butter,Diapers"
T102,"Bread,Butter,Nutella"
T103,"Milk,Diapers,Beer,Eggs"
T104,"Milk,Bread,Butter,Diapers,Beer"
T105,"Bread,Butter"
T106,"Milk,Diapers,Beer"
T107,"Milk,Bread,Butter"
T108,"Bread,Nutella,Coffee"
T109,"Milk,Diapers,Beer,Eggs"
T110,"Bread,Butter,Diapers"
T111,"Milk,Bread,Butter,Coffee"
T112,"Milk,Diapers,Beer"
T113,"Bread,Butter,Nutella,Coffee"
T114,"Milk,Bread,Diapers,Beer"
T115,"Milk,Bread,Butter,Diapers"
T116,"Bread,Butter,Coffee"
T117,"Milk,Diapers,Beer,Eggs"
T118,"Milk,Bread,Butter,Nutella"
T119,"Bread,Butter,Diapers"
T120,"Milk,Diapers,Beer"
"""

# --- HELPER FUNCTIONS ---
def generate_strategy(lift, conf, supp):
    """Translates metrics into plain-English manager directives."""
    if lift > 1.5 and conf >= 0.8:
        return "📦 BUNDLE — place these on the same shelf, endcap, or combo deal."
    elif lift > 1.2 and supp > 0.35:
        return "🚶 SPREAD OUT — put these on opposite ends so shoppers walk the aisle."
    elif conf >= 0.9:
        return "🏷️ PROMOTE — discount the first item to pull sales of the second."
    else:
        return "✅ STABLE — current placement is fine, no action needed."

def process_data(df, min_supp):
    """Runs Apriori, filters to 1:1 items, and deduplicates mirror rules."""
    # Clean and encode items
    transactions = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Run algorithms
    freq_items = apriori(df_encoded, min_support=min_supp, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame()
        
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
    if rules.empty:
        return pd.DataFrame()
    
    # Filter to 1-to-1 relationships only
    rules = rules[
        (rules['antecedents'].apply(len) == 1) & 
        (rules['consequents'].apply(len) == 1)
    ].copy()
    
    if rules.empty:
        return pd.DataFrame()
        
    # Extract plain string names
    rules['Item_A'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Item_B'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    # Deduplicate mirror-images (Keep the direction with highest confidence)
    # Create an unordered pair set for deduplication
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Item_A'], row['Item_B']]), axis=1)
    
    # Sort by confidence so drop_duplicates keeps the strongest directional instruction
    rules = rules.sort_values(by='confidence', ascending=False)
    rules = rules.drop_duplicates(subset=['Pair_ID'], keep='first')
    
    # Sort final view by how often they occur (support)
    rules = rules.sort_values(by='support', ascending=False)
    
    # Generate instructions
    rules['Action'] = rules.apply(lambda row: generate_strategy(row['lift'], row['confidence'], row['support']), axis=1)
    
    return rules

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📥 Data Upload")
    uploaded_file = st.file_uploader("Upload Monthly Receipts (CSV)", type="csv")
    
    st.markdown("""
    <div style='background-color: #e6f2ff; padding: 10px; border-radius: 5px; font-size: 13px; color: #004c8c; margin-bottom: 20px;'>
        <strong>Expected Columns:</strong> <code>Transaction_ID</code>, <code>Items</code> (comma-separated).<br>
        <em>Optional:</em> <code>Price</code> or <code>Amount</code>.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### ⚙️ Settings")
    sensitivity = st.slider(
        "Rule Sensitivity", 
        min_value=0.05, max_value=0.50, value=0.15, step=0.05,
        help="Lower this if you aren't seeing enough rules. Raise it to only see the most common pairings."
    )

# --- DATA LOADING & KPI CALCULATION ---
is_sample_data = False
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'Transaction_ID' not in df.columns or 'Items' not in df.columns:
            st.error("CSV must contain 'Transaction_ID' and 'Items' columns.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    is_sample_data = True
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))

# Calculate KPIs
total_txns = len(df)
all_items = [item.strip() for sublist in df['Items'].astype(str).str.split(',') for item in sublist]
avg_basket = round(len(all_items) / total_txns, 1) if total_txns > 0 else 0
best_seller = pd.Series(all_items).mode()[0] if all_items else "N/A"

# Determine 4th KPI (Sales vs Unique Products)
price_col = next((col for col in df.columns if col.lower() in ['price', 'amount']), None)
if price_col:
    kpi4_title = "💰 Total Sales"
    kpi4_value = f"${df[price_col].sum():,.2f}"
else:
    kpi4_title = "📦 Unique Products Sold"
    kpi4_value = f"{len(set(all_items))}"

# --- MAIN DASHBOARD UI ---

# 1. Header Banner
st.markdown("""
<div class="main-banner">
    <h1>🛒 Supercenter Strategy Dashboard</h1>
    <p>One page. What happened this month, and what to do about it.</p>
</div>
""", unsafe_allow_html=True)

if is_sample_data:
    st.info("ℹ️ **Showing Sample Data.** Upload your own store's CSV in the sidebar to see your metrics.")

# 2. KPI Cards
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-title">🧾 Transactions</div>
        <div class="kpi-value">{total_txns:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">🛍️ Avg. Basket Size</div>
        <div class="kpi-value">{avg_basket} items</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">⭐ Best-Selling Product</div>
        <div class="kpi-value">{best_seller}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">{kpi4_title}</div>
        <div class="kpi-value">{kpi4_value}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Process rules
rules_df = process_data(df, sensitivity)

if rules_df.empty:
    st.warning("No strong product pairings found at this sensitivity level. Try lowering the 'Rule Sensitivity' slider in the sidebar, or upload a larger dataset.")
else:
    # 3. Chart: Top 5 Pairings
    st.markdown("### 📊 Top 5 Product Pairings")
    st.caption("How often these items show up together in the same cart.")
    
    chart_data = rules_df.head(5).copy()
    chart_data['Pair Name'] = chart_data['Item_A'] + " & " + chart_data['Item_B']
    chart_data['% of Total Checkouts'] = chart_data['support'] * 100
    
    bar_chart = alt.Chart(chart_data).mark_bar(color='#0071CE', cornerRadiusEnd=4, height=30).encode(
        x=alt.X('% of Total Checkouts:Q', title='Percentage of Total Checkouts'),
        y=alt.Y('Pair Name:N', sort='-x', title='', axis=alt.Axis(labelFontSize=13)),
        tooltip=[alt.Tooltip('Pair Name', title='Pairing'), alt.Tooltip('% of Total Checkouts', format='.1f', title='Checkouts %')]
    ).properties(height=250)
    
    st.altair_chart(bar_chart, use_container_width=True)

    # 4. Key Takeaways
    st.markdown("### 💡 Key Takeaways")
    takeaways = ""
    for i, row in rules_df.head(3).iterrows():
        pct = round(row['confidence'] * 100)
        item_a = row['Item_A']
        item_b = row['Item_B']
        action_text = row['Action'].split('—')[-1].strip().lower() # Extracts just the plain english action
        takeaways += f"<li>Customers who buy <strong>{item_a}</strong> also buy <strong>{item_b}</strong> {pct}% of the time — you should {action_text}</li>"
    
    st.markdown(f"""
    <div class="takeaways-box">
        <ul>{takeaways}</ul>
    </div>
    """, unsafe_allow_html=True)

    # 5. Action Plan Table
    st.markdown("### 📋 Action Plan")
    
    display_cols = rules_df[['Item_A', 'Item_B', 'Action']].copy()
    display_cols.columns = ['If they buy...', 'They usually also buy...', 'What you should do']
    st.dataframe(display_cols, hide_index=True, use_container_width=True)

    # 6. Collapsed Analysts Section
    st.write("")
    with st.expander("🔍 See the underlying numbers (for analysts)"):
        st.markdown("""
        * **Support %:** How often this specific pair of items appears in *all* store transactions.
        * **Confidence %:** When a customer buys Item A, the probability they will also buy Item B.
        * **Lift:** How much more likely these items are bought together compared to being bought completely independently (Lift > 1 means a positive correlation).
        """)
        
        analyst_df = rules_df[['Item_A', 'Item_B', 'support', 'confidence', 'lift']].copy()
        analyst_df['support'] = (analyst_df['support'] * 100).round(1).astype(str) + '%'
        analyst_df['confidence'] = (analyst_df['confidence'] * 100).round(1).astype(str) + '%'
        analyst_df['lift'] = analyst_df['lift'].round(2)
        analyst_df.columns = ['Item A', 'Item B', 'Support %', 'Confidence %', 'Lift']
        
        st.dataframe(analyst_df, hide_index=True, use_container_width=True)        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 40px;
    }
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        flex: 1;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .kpi-title {
        color: #666666;
        font-size: 15px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }
    .kpi-value {
        color: #0071CE;
        font-size: 32px;
        font-weight: bold;
    }
    
    /* Takeaways Box */
    .takeaways-box {
        background-color: #f4f8fb;
        border-left: 5px solid #FFC220;
        padding: 20px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 40px;
    }
    .takeaways-box h3 {
        margin-top: 0;
        color: #333;
        font-size: 18px;
    }
    .takeaways-box ul {
        margin-bottom: 0;
        font-size: 16px;
        color: #444;
    }
    .takeaways-box li {
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- BUILT-IN SAMPLE DATA ---
SAMPLE_CSV = """Transaction_ID,Items
T101,"Milk,Bread,Butter,Diapers"
T102,"Bread,Butter,Nutella"
T103,"Milk,Diapers,Beer,Eggs"
T104,"Milk,Bread,Butter,Diapers,Beer"
T105,"Bread,Butter"
T106,"Milk,Diapers,Beer"
T107,"Milk,Bread,Butter"
T108,"Bread,Nutella,Coffee"
T109,"Milk,Diapers,Beer,Eggs"
T110,"Bread,Butter,Diapers"
T111,"Milk,Bread,Butter,Coffee"
T112,"Milk,Diapers,Beer"
T113,"Bread,Butter,Nutella,Coffee"
T114,"Milk,Bread,Diapers,Beer"
T115,"Milk,Bread,Butter,Diapers"
T116,"Bread,Butter,Coffee"
T117,"Milk,Diapers,Beer,Eggs"
T118,"Milk,Bread,Butter,Nutella"
T119,"Bread,Butter,Diapers"
T120,"Milk,Diapers,Beer"
"""

# --- HELPER FUNCTIONS ---
def generate_strategy(lift, conf, supp):
    """Translates metrics into plain-English manager directives."""
    if lift > 1.5 and conf >= 0.8:
        return "📦 BUNDLE — place these on the same shelf, endcap, or combo deal."
    elif lift > 1.2 and supp > 0.35:
        return "🚶 SPREAD OUT — put these on opposite ends so shoppers walk the aisle."
    elif conf >= 0.9:
        return "🏷️ PROMOTE — discount the first item to pull sales of the second."
    else:
        return "✅ STABLE — current placement is fine, no action needed."

def process_data(df, min_supp):
    """Runs Apriori, filters to 1:1 items, and deduplicates mirror rules."""
    # Clean and encode items
    transactions = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Run algorithms
    freq_items = apriori(df_encoded, min_support=min_supp, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame()
        
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
    if rules.empty:
        return pd.DataFrame()
    
    # Filter to 1-to-1 relationships only
    rules = rules[
        (rules['antecedents'].apply(len) == 1) & 
        (rules['consequents'].apply(len) == 1)
    ].copy()
    
    if rules.empty:
        return pd.DataFrame()
        
    # Extract plain string names
    rules['Item_A'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Item_B'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    # Deduplicate mirror-images (Keep the direction with highest confidence)
    # Create an unordered pair set for deduplication
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Item_A'], row['Item_B']]), axis=1)
    
    # Sort by confidence so drop_duplicates keeps the strongest directional instruction
    rules = rules.sort_values(by='confidence', ascending=False)
    rules = rules.drop_duplicates(subset=['Pair_ID'], keep='first')
    
    # Sort final view by how often they occur (support)
    rules = rules.sort_values(by='support', ascending=False)
    
    # Generate instructions
    rules['Action'] = rules.apply(lambda row: generate_strategy(row['lift'], row['confidence'], row['support']), axis=1)
    
    return rules

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📥 Data Upload")
    uploaded_file = st.file_uploader("Upload Monthly Receipts (CSV)", type="csv")
    
    st.markdown("""
    <div style='background-color: #e6f2ff; padding: 10px; border-radius: 5px; font-size: 13px; color: #004c8c; margin-bottom: 20px;'>
        <strong>Expected Columns:</strong> <code>Transaction_ID</code>, <code>Items</code> (comma-separated).<br>
        <em>Optional:</em> <code>Price</code> or <code>Amount</code>.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### ⚙️ Settings")
    sensitivity = st.slider(
        "Rule Sensitivity", 
        min_value=0.05, max_value=0.50, value=0.15, step=0.05,
        help="Lower this if you aren't seeing enough rules. Raise it to only see the most common pairings."
    )

# --- DATA LOADING & KPI CALCULATION ---
is_sample_data = False
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'Transaction_ID' not in df.columns or 'Items' not in df.columns:
            st.error("CSV must contain 'Transaction_ID' and 'Items' columns.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    is_sample_data = True
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))

# Calculate KPIs
total_txns = len(df)
all_items = [item.strip() for sublist in df['Items'].astype(str).str.split(',') for item in sublist]
avg_basket = round(len(all_items) / total_txns, 1) if total_txns > 0 else 0
best_seller = pd.Series(all_items).mode()[0] if all_items else "N/A"

# Determine 4th KPI (Sales vs Unique Products)
price_col = next((col for col in df.columns if col.lower() in ['price', 'amount']), None)
if price_col:
    kpi4_title = "💰 Total Sales"
    kpi4_value = f"${df[price_col].sum():,.2f}"
else:
    kpi4_title = "📦 Unique Products Sold"
    kpi4_value = f"{len(set(all_items))}"

# --- MAIN DASHBOARD UI ---

# 1. Header Banner
st.markdown("""
<div class="main-banner">
    <h1>🛒 Supercenter Strategy Dashboard</h1>
    <p>One page. What happened this month, and what to do about it.</p>
</div>
""", unsafe_allow_html=True)

if is_sample_data:
    st.info("ℹ️ **Showing Sample Data.** Upload your own store's CSV in the sidebar to see your metrics.")

# 2. KPI Cards
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-title">🧾 Transactions</div>
        <div class="kpi-value">{total_txns:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">🛍️ Avg. Basket Size</div>
        <div class="kpi-value">{avg_basket} items</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">⭐ Best-Selling Product</div>
        <div class="kpi-value">{best_seller}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">{kpi4_title}</div>
        <div class="kpi-value">{kpi4_value}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Process rules
rules_df = process_data(df, sensitivity)

if rules_df.empty:
    st.warning("No strong product pairings found at this sensitivity level. Try lowering the 'Rule Sensitivity' slider in the sidebar, or upload a larger dataset.")
else:
    # 3. Chart: Top 5 Pairings
    st.markdown("### 📊 Top 5 Product Pairings")
    st.caption("How often these items show up together in the same cart.")
    
    chart_data = rules_df.head(5).copy()
    chart_data['Pair Name'] = chart_data['Item_A'] + " & " + chart_data['Item_B']
    chart_data['% of Total Checkouts'] = chart_data['support'] * 100
    
    bar_chart = alt.Chart(chart_data).mark_bar(color='#0071CE', cornerRadiusEnd=4, height=30).encode(
        x=alt.X('% of Total Checkouts:Q', title='Percentage of Total Checkouts'),
        y=alt.Y('Pair Name:N', sort='-x', title='', axis=alt.Axis(labelFontSize=13)),
        tooltip=[alt.Tooltip('Pair Name', title='Pairing'), alt.Tooltip('% of Total Checkouts', format='.1f', title='Checkouts %')]
    ).properties(height=250)
    
    st.altair_chart(bar_chart, use_container_width=True)

    # 4. Key Takeaways
    st.markdown("### 💡 Key Takeaways")
    takeaways = ""
    for i, row in rules_df.head(3).iterrows():
        pct = round(row['confidence'] * 100)
        item_a = row['Item_A']
        item_b = row['Item_B']
        action_text = row['Action'].split('—')[-1].strip().lower() # Extracts just the plain english action
        takeaways += f"<li>Customers who buy <strong>{item_a}</strong> also buy <strong>{item_b}</strong> {pct}% of the time — you should {action_text}</li>"
    
    st.markdown(f"""
    <div class="takeaways-box">
        <ul>{takeaways}</ul>
    </div>
    """, unsafe_allow_html=True)

    # 5. Action Plan Table
    st.markdown("### 📋 Action Plan")
    
    display_cols = rules_df[['Item_A', 'Item_B', 'Action']].copy()
    display_cols.columns = ['If they buy...', 'They usually also buy...', 'What you should do']
    st.dataframe(display_cols, hide_index=True, use_container_width=True)

    # 6. Collapsed Analysts Section
    st.write("")
    with st.expander("🔍 See the underlying numbers (for analysts)"):
        st.markdown("""
        * **Support %:** How often this specific pair of items appears in *all* store transactions.
        * **Confidence %:** When a customer buys Item A, the probability they will also buy Item B.
        * **Lift:** How much more likely these items are bought together compared to being bought completely independently (Lift > 1 means a positive correlation).
        """)
        
        analyst_df = rules_df[['Item_A', 'Item_B', 'support', 'confidence', 'lift']].copy()
        analyst_df['support'] = (analyst_df['support'] * 100).round(1).astype(str) + '%'
        analyst_df['confidence'] = (analyst_df['confidence'] * 100).round(1).astype(str) + '%'
        analyst_df['lift'] = analyst_df['lift'].round(2)
        analyst_df.columns = ['Item A', 'Item B', 'Support %', 'Confidence %', 'Lift']
        
        st.dataframe(analyst_df, hide_index=True, use_container_width=True)        <h1 style="color:white; margin:0; font-family:Arial, sans-serif;">🛒 Supercenter Strategy Dashboard</h1>
        <p style="color:#FFC220; margin:6px 0 0 0; font-size:18px; font-weight:600;">
            One page. What happened this month, and what to do about it.
        </p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# 2. SAMPLE DATA (so the dashboard works before any upload)
# =========================================================
SAMPLE_CSV = """Transaction_ID,Items
T101,"Milk,Bread,Butter,Diapers"
T102,"Bread,Butter,Nutella"
T103,"Milk,Diapers,Beer,Eggs"
T104,"Milk,Bread,Butter,Diapers,Beer"
T105,"Bread,Butter"
T106,"Milk,Diapers,Beer"
T107,"Milk,Bread,Butter"
T108,"Bread,Nutella,Coffee"
T109,"Milk,Diapers,Beer,Eggs"
T110,"Bread,Butter,Diapers"
T111,"Milk,Bread,Butter,Coffee"
T112,"Milk,Diapers,Beer"
T113,"Bread,Butter,Nutella,Coffee"
T114,"Milk,Bread,Diapers,Beer"
T115,"Milk,Bread,Butter,Diapers"
T116,"Bread,Butter,Coffee"
T117,"Milk,Diapers,Beer,Eggs"
T118,"Milk,Bread,Butter,Nutella"
T119,"Bread,Butter,Diapers"
T120,"Milk,Diapers,Beer"
"""

# =========================================================
# 3. SIDEBAR — DATA UPLOAD
# =========================================================
st.sidebar.markdown("### 📥 Monthly Data Update")
uploaded_file = st.sidebar.file_uploader("Upload this month's receipts (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    using_sample = False
else:
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))
    using_sample = True
    st.sidebar.info("Showing sample data. Upload a CSV to see your own store's numbers.")

st.sidebar.divider()
st.sidebar.caption("Expected columns: **Transaction_ID**, **Items** (comma-separated). "
                    "Optional: **Price** or **Amount**, **Date**.")
st.sidebar.divider()
st.sidebar.caption("For the bright layout: click ⋮ (top right) → Settings → Theme → Light.")

# =========================================================
# 4. CLEAN THE DATA
# =========================================================
required_cols = {"Transaction_ID", "Items"}
if not required_cols.issubset(set(df.columns)):
    st.error("Your file needs at least these two columns: 'Transaction_ID' and 'Items'.")
    st.stop()

df = df.dropna(subset=["Items"]).copy()
df["Items"] = df["Items"].astype(str)
transactions = df["Items"].str.split(",").apply(lambda items: [i.strip() for i in items if i.strip()])

# =========================================================
# 5. TOP KPI CARDS
# =========================================================
total_transactions = df["Transaction_ID"].nunique()
basket_sizes = transactions.apply(len)
avg_basket_size = round(basket_sizes.mean(), 1) if len(basket_sizes) else 0
all_items = [item for basket in transactions for item in basket]
best_seller = pd.Series(all_items).value_counts().idxmax() if all_items else "—"

# Optional revenue column (Price or Amount), if present
revenue_col = next((c for c in ["Price", "Amount", "Total", "Sales"] if c in df.columns), None)
if revenue_col:
    total_sales = df[revenue_col].sum()
    sales_display = f"₹{total_sales:,.0f}"
else:
    sales_display = f"{len(set(all_items))} products"

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi_data = [
    ("🧾 Transactions", f"{total_transactions:,}"),
    ("🛍️ Avg. Basket Size", f"{avg_basket_size} items"),
    ("⭐ Best-Selling Product", best_seller),
    ("💰 Total Sales" if revenue_col else "📦 Products Sold", sales_display),
]
for col, (label, value) in zip([kpi1, kpi2, kpi3, kpi4], kpi_data):
    with col:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
        """, unsafe_allow_html=True)

st.write("")
st.divider()

# =========================================================
# 6. MARKET BASKET ANALYSIS (support / confidence / lift under the hood)
# =========================================================
def generate_strategy(row):
    """Turns the math into a one-line instruction a manager can act on today."""
    if row["lift"] > 1.5 and row["confidence"] >= 0.8:
        return "📦 BUNDLE — place these on the same shelf, endcap, or combo deal."
    elif row["lift"] > 1.2 and row["support"] > 0.35:
        return "🚶 SPREAD OUT — put these on opposite ends so shoppers walk the aisle."
    elif row["confidence"] >= 0.9:
        return "🏷️ PROMOTE — discount the first item to pull sales of the second."
    return "✅ STABLE — current placement is fine, no action needed."


min_support = st.sidebar.slider(
    "Sensitivity (advanced)", 0.05, 0.5, 0.15, 0.05,
    help="Lower = catches more/weaker patterns. Higher = only the strongest, most frequent pairings."
)

te = TransactionEncoder()
te_ary = te.fit(transactions.tolist()).transform(transactions.tolist())
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

if frequent_itemsets.empty:
    st.warning("Not enough repeat patterns found yet. Try lowering the sensitivity slider in the sidebar, "
               "or upload a larger set of transactions.")
    st.stop()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Keep only simple, explainable one-item -> one-item rules (what a manager can actually act on)
rules = rules[(rules["antecedents"].apply(len) == 1) & (rules["consequents"].apply(len) == 1)].copy()

if rules.empty:
    st.warning("No strong, simple product pairings found in this data yet.")
    st.stop()

rules["A"] = rules["antecedents"].apply(lambda x: list(x)[0])
rules["B"] = rules["consequents"].apply(lambda x: list(x)[0])

# De-duplicate mirror pairs (Bread->Butter and Butter->Bread are the same pairing).
# Keep the direction that gives the manager the clearer instruction: highest confidence.
rules["pair_key"] = rules.apply(lambda r: tuple(sorted([r["A"], r["B"]])), axis=1)
rules = rules.sort_values("confidence", ascending=False).drop_duplicates(subset="pair_key", keep="first")
rules["Manager_Action"] = rules.apply(generate_strategy, axis=1)
rules = rules.sort_values("support", ascending=False).reset_index(drop=True)

# =========================================================
# 7. ONE CHART — Top 5 pairings, in plain language
# =========================================================
st.markdown("### 📊 Top 5 Product Pairings")
st.caption("How often these items show up together in the same cart.")

chart_data = rules.head(5).copy()
chart_data["Pairing"] = chart_data["A"] + " & " + chart_data["B"]
chart_data["% of All Baskets"] = (chart_data["support"] * 100).round(1)

bar_chart = (
    alt.Chart(chart_data)
    .mark_bar(color="#0071CE", cornerRadiusEnd=4)
    .encode(
        x=alt.X("% of All Baskets:Q", title="Percentage of Total Checkouts"),
        y=alt.Y("Pairing:N", sort="-x", title=None),
        tooltip=["Pairing", "% of All Baskets"],
    )
    .properties(height=260)
)
st.altair_chart(bar_chart, use_container_width=True)

st.divider()

# =========================================================
# 8. KEY TAKEAWAYS — plain-English, auto-generated
# =========================================================
st.markdown("### 💡 Key Takeaways")

top_rules = rules.head(3)
bullets = []
for _, r in top_rules.iterrows():
    pct = round(r["confidence"] * 100)
    bullets.append(f"Customers who buy **{r['A']}** buy **{r['B']}** {pct}% of the time — "
                    f"{r['Manager_Action'].split('—')[1].strip().lower()}")

st.markdown(
    f'<div class="takeaway-box">' + "<br><br>".join(f"• {b}" for b in bullets) + "</div>",
    unsafe_allow_html=True,
)

st.write("")
st.divider()

# =========================================================
# 9. ACTION PLAN TABLE — the main deliverable
# =========================================================
st.markdown("### 📋 Action Plan")
st.caption("What to do this month, based on how customers actually shop.")

display_df = rules[["A", "B", "Manager_Action"]].copy()
display_df.columns = ["If they buy...", "They usually also buy...", "What you should do"]
st.dataframe(display_df, hide_index=True, use_container_width=True)

with st.expander("🔍 See the underlying numbers (for analysts)"):
    detail_df = rules[["A", "B", "support", "confidence", "lift"]].copy()
    detail_df.columns = ["Item A", "Item B", "Support", "Confidence", "Lift"]
    detail_df["Support"] = (detail_df["Support"] * 100).round(1).astype(str) + "%"
    detail_df["Confidence"] = (detail_df["Confidence"] * 100).round(1).astype(str) + "%"
    detail_df["Lift"] = detail_df["Lift"].round(2)
    st.dataframe(detail_df, hide_index=True, use_container_width=True)
    st.caption(
        "**Support** = how common this pairing is across all carts. "
        "**Confidence** = how often buying Item A leads to buying Item B. "
        "**Lift** = how much more likely this pairing is than pure chance (above 1 = a real pattern)."
    )

if using_sample:
    st.info("👆 This is sample data. Upload your own monthly CSV from the sidebar to replace it.")
