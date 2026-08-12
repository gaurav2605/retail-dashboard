import streamlit as st
import pandas as pd
import altair as alt
import io
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# =========================================================
# 1. PAGE SETUP
# =========================================================
st.set_page_config(page_title="Store Manager Dashboard", page_icon="🛒", layout="wide")

st.markdown("""
    <style>
    .kpi-card {
        background-color: #F5F7FA;
        border-radius: 12px;
        padding: 22px 18px;
        text-align: center;
        border: 1px solid #E3E7EC;
    }
    .kpi-value { font-size: 34px; font-weight: 800; color: #0071CE; margin: 4px 0 0 0; }
    .kpi-label { font-size: 14px; color: #5A6472; font-weight: 600; letter-spacing: .3px; }
    .takeaway-box {
        background-color: #FFF8E6;
        border-left: 6px solid #FFC220;
        border-radius: 8px;
        padding: 18px 22px;
        font-size: 16px;
        line-height: 1.7;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="background-color:#0071CE; padding:26px 30px; border-radius:10px; margin-bottom:24px;">
        <h1 style="color:white; margin:0; font-family:Arial, sans-serif;">🛒 Supercenter Strategy Dashboard</h1>
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
