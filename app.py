import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ─────────────────────────────────────
# Loading and Training Model
# ─────────────────────────────────────
df = pd.read_csv('house_prices_cleaned.csv')

X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ─────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")

# ─────────────────────────────────────
# Tabs
# ─────────────────────────────────────
tab1, tab2 = st.tabs(["🔮 Predict", "📊 Analysis"])

# ═════════════════════════════════════
# TAB 1 — PREDICT
# ═════════════════════════════════════
with tab1:
    st.subheader("Enter House Details")
    st.write("Fill in the details below to predict the house price!")
    st.divider()

    # Session State for Reset Button
    if 'sqft' not in st.session_state:
        st.session_state['sqft'] = 1500
    if 'bedrooms' not in st.session_state:
        st.session_state['bedrooms'] = 3
    if 'bathrooms' not in st.session_state:
        st.session_state['bathrooms'] = 2

    # Reset Button
    if st.button("🔄 Reset"):
        st.session_state['sqft'] = 1500
        st.session_state['bedrooms'] = 3
        st.session_state['bathrooms'] = 2

    # Input Fields
    sqft      = st.number_input("📐 Square Footage", min_value=100, max_value=10000, key='sqft')
    bedrooms  = st.number_input("🛏️ Number of Bedrooms", min_value=0, max_value=10, key='bedrooms')
    bathrooms = st.number_input("🚿 Number of Bathrooms", min_value=0, max_value=10, key='bathrooms')

    st.divider()

    if st.button("🔮 Predict Price"):

        input_data      = np.array([[sqft, bedrooms, bathrooms]])
        predicted_price = model.predict(input_data)[0]

        # Save to session state (notebook)
        st.session_state.predicted_price = predicted_price
        st.session_state.input_sqft      = sqft
        st.session_state.input_bedrooms  = bedrooms
        st.session_state.input_bathrooms = bathrooms

        # Predicted Price
        st.success(f"💰 Predicted House Price: ${predicted_price:,.2f}")
        st.balloons()
        st.divider()

        # Price Category
        st.subheader("🏷️ Price Category")
        if predicted_price < 150000:
            st.success("🟢 Budget Friendly")
        elif predicted_price < 300000:
            st.warning("🟡 Mid Range")
        else:
            st.error("🔴 Luxury")
        st.divider()

        # Price Per Sqft
        st.subheader("📐 Price Per Square Foot")
        price_per_sqft = predicted_price / sqft
        st.info(f"💵 ${price_per_sqft:,.2f} per sqft")
        st.divider()

        # Price Range
        st.subheader("📊 Expected Price Range")
        min_price = predicted_price - rmse
        max_price = predicted_price + rmse
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Minimum Price", f"${min_price:,.2f}")
        with col2:
            st.metric("Predicted Price", f"${predicted_price:,.2f}")
        with col3:
            st.metric("Maximum Price", f"${max_price:,.2f}")
        st.divider()

        # Affordability Meter
        st.subheader("📏 Affordability Meter")
        min_dataset_price = df['price'].min()
        max_dataset_price = df['price'].max()
        affordability = (predicted_price - min_dataset_price) / (max_dataset_price - min_dataset_price)
        affordability_clamped = min(max(float(affordability), 0.0), 1.0)
        bar_percent = affordability_clamped * 100
        st.markdown(f"""
            <div style="
                background: #E3F2FD;
                border-radius: 10px;
                height: 20px;
                width: 100%;
                margin: 10px 0;">
                <div style="
                    background: linear-gradient(90deg, #1A73E8, #00BFFF);
                    width: {bar_percent}%;
                    height: 20px;
                    border-radius: 10px;
                    transition: width 0.5s ease;">
                </div>
            </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"💚 Cheapest: ${min_dataset_price:,.0f}")
        with col2:
            st.write(f"📍 Your House: {affordability_clamped*100:.1f}%")
        with col3:
            st.write(f"🔴 Most Expensive: ${max_dataset_price:,.0f}")
        st.divider()

        # Similar Houses Table
        st.subheader("🏠 Similar Houses from Dataset")
        df['difference'] = abs(df['sqft'] - sqft) + abs(df['bedrooms'] - bedrooms) * 100 + abs(df['bathrooms'] - bathrooms) * 100
        similar_houses   = df.nsmallest(5, 'difference')[['sqft', 'bedrooms', 'bathrooms', 'price']].reset_index(drop=True)
        similar_houses.index += 1
        similar_houses.columns = ['Square Footage', 'Bedrooms', 'Bathrooms', 'Actual Price ($)']
        st.table(similar_houses)

# ═════════════════════════════════════
# TAB 2 — ANALYSIS
# ═════════════════════════════════════
with tab2:
    st.subheader("📊 Data Analysis & Visualizations")
    st.divider()

    if 'predicted_price' not in st.session_state:
        st.warning("⚠️ Please go to Predict tab first and predict a price!")

    else:
        predicted_price  = st.session_state.predicted_price
        input_sqft       = st.session_state.input_sqft
        input_bedrooms   = st.session_state.input_bedrooms
        input_bathrooms  = st.session_state.input_bathrooms

        # ── Graph 1 — Price Distribution ──
        st.subheader("1️⃣ Price Distribution")
        st.write("Red line shows where YOUR predicted price falls among all houses!")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.histplot(df['price'], kde=True, color='blue', ax=ax1)
        ax1.axvline(x=predicted_price, color='red',
                    linewidth=2, linestyle='--',
                    label=f'Your House: ${predicted_price:,.0f}')
        ax1.set_xlabel("Price ($)")
        ax1.set_ylabel("Number of Houses")
        ax1.set_title("Distribution of House Prices")
        ax1.legend()
        st.pyplot(fig1)
        st.divider()

        # ── Graph 2 — Correlation Heatmap ──
        st.subheader("2️⃣ Correlation Heatmap")
        st.write("Shows how strongly each feature is related to house price!")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[['sqft', 'bedrooms', 'bathrooms', 'price']].corr(),
                    annot=True, cmap='coolwarm', fmt='.2f',
                    linewidths=0.5, ax=ax2)
        ax2.set_title("Correlation Heatmap")
        st.pyplot(fig2)
        st.divider()

        # ── Graph 3 — Feature Importance ──
        st.subheader("3️⃣ Feature Importance Chart")
        st.write("Shows how much each feature contributed to YOUR house price!")
        contributions = [
            abs(model.coef_[0] * input_sqft),
            abs(model.coef_[1] * input_bedrooms),
            abs(model.coef_[2] * input_bathrooms)
        ]
        features = ['Square Footage', 'Bedrooms', 'Bathrooms']
        colors   = ['blue' if c != max(contributions) else 'red' for c in contributions]
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.bar(features, contributions, color=colors)
        ax3.set_xlabel("Features")
        ax3.set_ylabel("Contribution to Price ($)")
        ax3.set_title("Feature Contribution for YOUR House")
        st.pyplot(fig3)
        st.write("🔴 Red bar = most important feature for your house!")
        st.divider()

        # ── Graph 4 — Scatter Plot ──
        st.subheader("4️⃣ Scatter Plot — Sqft vs Price")
        st.write("Red dot shows YOUR house among all 1460 real houses!")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.scatter(df['sqft'], df['price'],
                    color='blue', alpha=0.4, label='All Houses')
        ax4.scatter(input_sqft, predicted_price,
                    color='red', s=200, zorder=5, label='Your House')
        ax4.set_xlabel("Square Footage")
        ax4.set_ylabel("Price ($)")
        ax4.set_title("Square Footage vs Price")
        ax4.legend()
        st.pyplot(fig4)
        st.divider()

        # ── Graph 5 — Price Trend Line ──
        st.subheader("5️⃣ Price Trend Line")
        st.write("Red dot shows where YOUR house sits on the price trend!")
        fig5, ax5    = plt.subplots(figsize=(10, 5))
        sqft_range   = np.linspace(df['sqft'].min(), df['sqft'].max(), 100)
        avg_bed      = df['bedrooms'].mean()
        avg_bath     = df['bathrooms'].mean()
        trend_data   = np.column_stack([sqft_range,
                                        np.full(100, avg_bed),
                                        np.full(100, avg_bath)])
        trend_prices = model.predict(trend_data)
        ax5.plot(sqft_range, trend_prices,
                 color='blue', linewidth=2, label='Price Trend')
        ax5.scatter(input_sqft, predicted_price,
                    color='red', s=200, zorder=5, label='Your House')
        ax5.set_xlabel("Square Footage")
        ax5.set_ylabel("Predicted Price ($)")
        ax5.set_title("Price Trend as Square Footage Increases")
        ax5.legend()
        st.pyplot(fig5)