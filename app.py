import streamlit as st
import pandas as pd
import pickle

# Set page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
    .stFileUploader>div>div>div>button {
        background-color: #0066cc;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Upload transaction data and detect fraudulent activity using a trained ML model.")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4139/4139981.png", width=100)
    st.header("Instructions")
    st.markdown("""
    1. Upload a `.csv` file with transaction data  
    2. Make sure the file has features: **Time, V1–V28, Amount**  
    3. The app will return predictions:  
       - **0** → Not Fraud  
       - **1** → Fraud
    """)

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your transaction CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Remove Class column if it exists
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)

    st.subheader("📄 Uploaded Data (first 5 rows)")
    st.dataframe(df.head())

    # Load model
    model = pickle.load(open("xgb_model.pkl", "rb"))

    # Predict
    predictions = model.predict(df)

    # Results
    st.subheader("🧠 Prediction Results")
    result_df = df.copy()
    result_df['Prediction'] = predictions
    st.write(result_df.head())

    fraud_count = sum(predictions)
    total = len(predictions)

    st.success(f"🔍 Found **{fraud_count}** fraudulent transactions out of **{total}**.")
