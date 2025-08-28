import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Title
st.title("💡 LightGBM – Bank Customer Churn Prediction")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Churn_Modelling.csv")
    return data

data = load_data()
st.subheader("📂 Dataset Preview")
st.write(data.head())

# Encode categorical features
label_enc = LabelEncoder()
data["Geography"] = label_enc.fit_transform(data["Geography"])
data["Gender"] = label_enc.fit_transform(data["Gender"])

# Features & Target
X = data.drop(["Exited", "RowNumber", "CustomerId", "Surname"], axis=1)
y = data["Exited"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1
}

model = lgb.train(params, train_data, num_boost_round=100)

# Predictions
y_pred = model.predict(X_test)
y_pred_class = [1 if prob > 0.5 else 0 for prob in y_pred]

# Results
acc = accuracy_score(y_test, y_pred_class)
st.subheader("📊 Model Performance")
st.write(f"✅ Accuracy: {acc:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred_class))
