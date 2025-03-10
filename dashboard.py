import streamlit as st
import pandas as pd
import plotly.express as px
import lightgbm as lgb
import pickle

# Load the trained model
model_path = "lgbm_model.pkl"  # Update with your model path
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Load preprocessed data
train_df = pd.read_csv("train_preprocessed.csv")
test_df = pd.read_csv("test_preprocessed.csv")

# Define features (Removed Date-related columns)
feature_cols = ["Dept", "Store", "CPI", "Unemployment", "Temperature", "Fuel_Price"]
X_train = train_df[feature_cols]
y_train = train_df["Weekly_Sales"]

# Streamlit UI
st.set_page_config(page_title="Walmart Sales Dashboard", layout="wide")
st.title("📊 Walmart Store Sales Dashboard")

# Key Insights
st.subheader("🔍 Key Sales Insights")

# Top Performing Stores
top_stores = train_df.groupby("Store")["Weekly_Sales"].sum().reset_index().sort_values(by="Weekly_Sales", ascending=False)
fig1 = px.bar(top_stores.head(5), x="Store", y="Weekly_Sales", title="🏪 Top 5 Best Performing Stores", color="Weekly_Sales", color_continuous_scale="viridis")
st.plotly_chart(fig1, use_container_width=True)

# Top Performing Departments
top_departments = train_df.groupby("Dept")["Weekly_Sales"].sum().reset_index().sort_values(by="Weekly_Sales", ascending=False)
fig2 = px.bar(top_departments.head(5), x="Dept", y="Weekly_Sales", title="🏢 Top 5 Best Performing Departments", color="Weekly_Sales", color_continuous_scale="blues")
st.plotly_chart(fig2, use_container_width=True)

# Sales Distribution
st.subheader("📉 Sales Distribution Across Stores")
fig4 = px.box(train_df, x="Store", y="Weekly_Sales", title="📦 Sales Variation Across Stores", color="Store")
st.plotly_chart(fig4, use_container_width=True)

# Feature Importance
st.subheader("📊 Feature Importance (LightGBM)")

# Get feature importances
importance = model.feature_importances_

# Ensure correct length match
if len(importance) == len(feature_cols):
    importance_df = pd.DataFrame({"Feature": feature_cols, "Importance": importance})
else:
    st.warning("Feature importance length mismatch! Adjusting dynamically.")
    feature_names = [f"Feature {i}" for i in range(len(importance))]  # Create generic names
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})

# Sort by importance
importance_df = importance_df.sort_values(by="Importance", ascending=True)

# Improved Feature Importance Visualization
fig3 = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="📌 Most Important Factors Affecting Sales", color="Importance", color_continuous_scale="reds")
st.plotly_chart(fig3, use_container_width=True)

# Recommendations
st.subheader("📌 Recommendations for Stakeholders")
st.markdown("""
- **📦 Inventory Management:** Focus on top-selling stores and departments to optimize stock.
- **💰 Pricing Strategy:** Adjust prices based on CPI and Unemployment trends.
- **📊 Marketing Focus:** Increase marketing efforts during high-sales periods.
- **🛒 Store Expansion:** Consider opening new locations in areas with high sales potential.
""")
