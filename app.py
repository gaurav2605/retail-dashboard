import streamlit as st
import pandas as pd
import os
import altair as alt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. WALMART-STYLE UI SETUP ---
st.set_page_config(page_title="Store Manager Dashboard", layout="wide")

# Custom Walmart-Blue Banner (Clean & Modern)
st.markdown("""
    <div style="background-color: #0071CE; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0; font-family: Arial, sans-serif;">🛒 Supercenter Strategy Dashboard</h1>
        <p style="color: #FFC220; margin: 0; font-size: 18px; font-weight: bold;">Smart placement and pricing recommendations</p>
    </div>
""", unsafe_allow_html=True)

# --- 2. MANAGER-FRIENDLY LOGIC ---
def generate_strategy(row):
    """Translates math into dead-simple manager instructions."""
    if row['lift'] > 1.5 and row['confidence'] >= 0.8:
        return "📦 BUNDLE: Put these items on the same shelf or endcap."
    elif row['lift'] > 1.2 and row['support'] > 0.4:
        return "🚶 FOOT TRAFFIC: Put these far apart so shoppers walk the whole aisle."
    elif row['confidence'] == 1.0:
        return "🏷️ PROMO: Discount the first item to drive full-price sales of the second."
    return "✅ STABLE: Keep current placement."

# --- 3. DATA LOADING ---
st.sidebar.markdown("### 📥 Monthly Data Update")
uploaded_file = st.sidebar.file_uploader("Upload new receipts (CSV)", type="csv")

data_path = "transactions.csv"
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = None
    st.info("Please upload a transaction CSV to see recommendations.")

# --- 4. ANALYTICS & VISUALS ---
if df is not None:
    transactions = df['Items'].str.split(',').tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules['Manager_Action'] = rules.apply(generate_strategy, axis=1)
        
        # Add a visual chart for the Top 5 Pairings
        st.markdown("### 📊 Top 5 Most Common Pairings")
        
        # Prepare data for the chart
        chart_data = rules.sort_values(by='support', ascending=False).head(5).copy()
        chart_data['Pairing'] = chart_data['antecedents'] + " & " + chart_data['consequents']
        chart_data['% of All Baskets'] = chart_data['support'] * 100
        
        # Create a modern blue bar chart
        bar_chart = alt.Chart(chart_data).mark_bar(color='#0071CE', cornerRadiusEnd=4).encode(
            x=alt.X('% of All Baskets:Q', title='Percentage of Total Checkouts'),
            y=alt.Y('Pairing:N', sort='-x', title='Product Pairing'),
            tooltip=['Pairing', '% of All Baskets']
        ).properties(height=300)
        
        st.altair_chart(bar_chart, use_container_width=True)
        st.divider()

        # Simplified Data Table
        st.markdown("### 📋 Action Plan")
        
        display_df = rules[['antecedents', 'consequents', 'Manager_Action']].copy()
        display_df.columns = [
            'If they buy...', 
            'They will also buy...', 
            'What you should do'
        ]
        
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No strong patterns found in this dataset.")

# Force Light Theme Instructions for the viewer
st.sidebar.divider()
st.sidebar.caption("To ensure the bright layout: Click the three dots (⋮) top right > Settings > Theme > Light.")
