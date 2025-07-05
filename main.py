import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Dự đoán giá đã giảm", layout="wide")

st.title("📉 Dự đoán giá giảm sản phẩm Amazon (Linear Regression)")

# Load data
url = "https://raw.githubusercontent.com/nguyenvudev20/25251325/refs/heads/main/amazon.csv"
df = pd.read_csv(url)

# Clean data
df['discounted_price'] = pd.to_numeric(df['discounted_price'].str.replace('₹', '').str.replace(',', ''), errors='coerce')
df['actual_price'] = pd.to_numeric(df['actual_price'].str.replace('₹', '').str.replace(',', ''), errors='coerce')
df['discount_percentage'] = pd.to_numeric(df['discount_percentage'].str.replace('%', ''), errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')
df.dropna(subset=['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count'], inplace=True)

# Hiển thị dữ liệu
if st.checkbox("📋 Hiển thị dữ liệu sau xử lý"):
    st.dataframe(df.head())

# Train model
X = df[['actual_price', 'discount_percentage', 'rating', 'rating_count']]
y = df['discounted_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Đánh giá mô hình Linear Regression")
st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
st.write(f"**R-squared Score (R²):** {r2:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Giá thật")
ax.set_ylabel("Giá dự đoán")
ax.set_title("So sánh giá thật và dự đoán")
st.pyplot(fig)

