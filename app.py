import streamlit as st
import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# --- UI Configuration ---
st.set_page_config(page_title="Retail Diagnostics", layout="wide")

# Modern, clean header
st.title("Retail Diagnostic Dashboard")
st.markdown("Automated layout and pricing strategies based on customer purchasing behavior.")
st.divider()

# --- Diagnostic Engine ---
def generate_strategy(row):
    """Translates raw statistical metrics into actionable business advice."""
    if row['lift'] > 1.5 and row['confidence'] >= 0.8:
        return "Bundle Opportunity: Cross-promote or place on adjacent shelves."
    elif row['lift'] > 1.2 and row['support'] > 0.4:
        return "Foot Traffic Driver: Separate these high-volume staples to encourage store exploration."
    elif row['confidence'] == 1.0:
        return "Discount Strategy: Discount the first item to guarantee full-price sales of the second."
    return "Maintain current placement."

# --- Data Loading ---
# Check if a file was uploaded by management; if not, use the default GitHub file
st.sidebar.markdown("### Update Data")
uploaded_file = st.sidebar.file_uploader("Upload new transaction CSV", type="csv")

data_path = "transactions.csv"
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = None
    st.info("Please upload a transaction CSV to generate diagnostics.")

# --- Analytics Pipeline ---
if df is not None:
    # 1. Clean and format the data
    transactions = df['Items'].str.split(',').tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 2. Run the algorithms
    frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    # 3. Apply the formatting and diagnostic logic
    if not rules.empty:
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        rules['Management_Action'] = rules.apply(generate_strategy, axis=1)
        
        # --- Dashboard Display ---
        # Top-level metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions Analyzed", len(transactions))
        col2.metric("Unique Products", len(te.columns_))
        col3.metric("Actionable Rules Found", len(rules))
        
        st.markdown("### Recommended Strategies")
        
        # Filter and rename columns for a simple, management-friendly view
        display_df = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'Management_Action']].copy()
        display_df['support'] = (display_df['support'] * 100).round(1).astype(str) + '%'
        display_df['confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
        display_df['lift'] = display_df['lift'].round(2)
        
        display_df.columns = [
            'When a customer buys...', 
            'They are likely to buy...', 
            'Basket %', 
            'Likelihood', 
            'Correlation', 
            'Diagnostic Strategy'
        ]
        
        # Render a clean data table
        st.dataframe(display_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No strong associations found in this dataset.")
