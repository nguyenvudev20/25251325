import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Dự đoán giá giảm Amazon", layout="wide")

st.title("📊 Dự đoán giá giảm sản phẩm Amazon với Linear Regression")

# Tải dữ liệu từ GitHub
url = "https://raw.githubusercontent.com/nguyenvudev20/daidaidiha/refs/heads/main/amazon.csv"
df = pd.read_csv(url)

# ============================
# 🧹 Tiền xử lý dữ liệu
# ============================
df['discounted_price'] = pd.to_numeric(df['discounted_price'].str.replace('₹', '').str.replace(',', ''), errors='coerce')
df['actual_price'] = pd.to_numeric(df['actual_price'].str.replace('₹', '').str.replace(',', ''), errors='coerce')
df['discount_percentage'] = pd.to_numeric(df['discount_percentage'].str.replace('%', ''), errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')

# Xử lý missing
df.dropna(subset=['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count'], inplace=True)

# Trích xuất danh mục chính
df['main_category'] = df['category'].str.split('|').str[0]

# ============================
# 📈 Phân tích & trực quan hóa
# ============================
st.subheader("📊 Phân tích dữ liệu & Trực quan hóa")

# Biểu đồ 1: Phân phối rating
fig1, ax1 = plt.subplots(figsize=(6, 4))
sns.histplot(df['rating'], bins=10, kde=True, ax=ax1)
ax1.set_title("1️⃣ Phân phối đánh giá sản phẩm")
ax1.set_xlabel("Điểm đánh giá (rating)")
st.pyplot(fig1)

# Biểu đồ 2: Tần suất mức giảm giá
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.histplot(df['discount_percentage'], bins=20, color='orange', ax=ax2)
ax2.set_title("2️⃣ Tần suất các mức giảm giá (%)")
ax2.set_xlabel("Phần trăm giảm giá")
st.pyplot(fig2)

# Biểu đồ 3: Tương quan actual vs discounted price
fig3, ax3 = plt.subplots(figsize=(6, 4))
sns.scatterplot(x='actual_price', y='discounted_price', data=df, alpha=0.6, ax=ax3)
ax3.set_title("3️⃣ Giá gốc vs Giá sau giảm")
st.pyplot(fig3)

# Biểu đồ 4: Top danh mục phổ biến
top_categories = df['main_category'].value_counts().head(10)
fig4, ax4 = plt.subplots(figsize=(6, 4))
sns.barplot(x=top_categories.values, y=top_categories.index, ax=ax4)
ax4.set_title("4️⃣ Top 10 danh mục sản phẩm phổ biến")
st.pyplot(fig4)

# Biểu đồ 5: Rating trung bình theo danh mục
rating_by_cat = df.groupby('main_category')['rating'].mean().sort_values(ascending=False).head(10)
fig5, ax5 = plt.subplots(figsize=(6, 4))
sns.barplot(x=rating_by_cat.values, y=rating_by_cat.index, palette='viridis', ax=ax5)
ax5.set_title("5️⃣ Trung bình đánh giá theo danh mục")
st.pyplot(fig5)

# ============================
# 🤖 Huấn luyện mô hình Linear Regression
# ============================
st.subheader("🤖 Dự đoán giá đã giảm bằng Linear Regression")

X = df[['actual_price', 'discount_percentage', 'rating', 'rating_count']]
y = df['discounted_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"📉 **Mean Squared Error (MSE):** {mse:,.2f}")
st.write(f"📈 **R-squared Score (R²):** {r2:.4f}")

# Biểu đồ so sánh giá thật và giá dự đoán
fig6, ax6 = plt.subplots(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax6)
ax6.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax6.set_xlabel("Giá thật")
ax6.set_ylabel("Giá dự đoán")
ax6.set_title("So sánh giá thật và giá dự đoán")
st.pyplot(fig6)
