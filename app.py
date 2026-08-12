import streamlit as st
import pandas as pd
import altair as alt
import io
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Supercenter Dashboard", layout="wide", initial_sidebar_state="expanded")

# Injecting CSS for a simple and modern design aesthetic
st.markdown("""
<style>
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

# --- 2. BUILT-IN SAMPLE DATA ---
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

# --- 3. HELPER FUNCTIONS ---
def generate_strategy(lift, conf, supp):
    if lift > 1.5 and conf >= 0.8:
        return "📦 BUNDLE — place these on the same shelf, endcap, or combo deal."
    elif lift > 1.2 and supp > 0.35:
        return "🚶 SPREAD OUT — put these on opposite ends so shoppers walk the aisle."
    elif conf >= 0.9:
        return "🏷️ PROMOTE — discount the first item to pull sales of the second."
    else:
        return "✅ STABLE — current placement is fine, no action needed."

def process_data(df, min_supp):
    transactions = df['Items'].astype(str).str.split(',').apply(lambda x: [i.strip() for i in x])
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    freq_items = apriori(df_encoded, min_support=min_supp, use_colnames=True)
    if freq_items.empty:
        return pd.DataFrame()
        
    rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
    if rules.empty:
        return pd.DataFrame()
    
    rules = rules[
        (rules['antecedents'].apply(len) == 1) & 
        (rules['consequents'].apply(len) == 1)
    ].copy()
    
    if rules.empty:
        return pd.DataFrame()
        
    rules['Item_A'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['Item_B'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    rules['Pair_ID'] = rules.apply(lambda row: frozenset([row['Item_A'], row['Item_B']]), axis=1)
    
    rules = rules.sort_values(by='confidence', ascending=False)
    rules = rules.drop_duplicates(subset=['Pair_ID'], keep='first')
    rules = rules.sort_values(by='support', ascending=False)
    
    rules['Action'] = rules.apply(lambda row: generate_strategy(row['lift'], row['confidence'], row['support']), axis=1)
    
    return rules

# --- 4. SIDEBAR ---
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

# --- 5. DATA LOADING & KPI CALCULATION ---
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

total_txns = len(df)
all_items = [item.strip() for sublist in df['Items'].astype(str).str.split(',') for item in sublist]
avg_basket = round(len(all_items) / total_txns, 1) if total_txns > 0 else 0
best_seller = pd.Series(all_items).mode()[0] if all_items else "N/A"

price_col = next((col for col in df.columns if col.lower() in ['price', 'amount']), None)
if price_col:
    kpi4_title = "💰 Total Sales"
    kpi4_value = f"${df[price_col].sum():,.2f}"
else:
    kpi4_title = "📦 Unique Products Sold"
    kpi4_value = f"{len(set(all_items))}"

# --- 6. MAIN DASHBOARD UI ---
st.markdown("""
<div class="main-banner">
    <h1>🛒 Supercenter Strategy Dashboard</h1>
    <p>One page. What happened this month, and what to do about it.</p>
</div>
""", unsafe_allow_html=True)

if is_sample_data:
    st.info("ℹ️ **Showing Sample Data.** Upload your own store's CSV in the sidebar to see your metrics.")

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

rules_df = process_data(df, sensitivity)

if rules_df.empty:
    st.warning("No strong product pairings found at this sensitivity level. Try lowering the 'Rule Sensitivity' slider in the sidebar, or upload a larger dataset.")
else:
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

    st.markdown("### 💡 Key Takeaways")
    takeaways = ""
    for i, row in rules_df.head(3).iterrows():
        pct = round(row['confidence'] * 100)
        item_a = row['Item_A']
        item_b = row['Item_B']
        action_text = row['Action'].split('—')[-1].strip().lower()
        takeaways += f"<li>Customers who buy <strong>{item_a}</strong> also buy <strong>{item_b}</strong> {pct}% of the time — you should {action_text}</li>"
    
    st.markdown(f"""
    <div class="takeaways-box">
        <ul>{takeaways}</ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Action Plan")
    
    display_cols = rules_df[['Item_A', 'Item_B', 'Action']].copy()
    display_cols.columns = ['If they buy...', 'They usually also buy...', 'What you should do']
    st.dataframe(display_cols, hide_index=True, use_container_width=True)

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
        
        st.dataframe(analyst_df, hide_index=True, use_container_width=True)
