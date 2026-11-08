import pickle
import pandas as pd
import streamlit as st

st.title("Supermarket Payment Prediction - Hein Ko Midterm Project")
st.write(
    "Enter the checkout details to estimate the customer's payment method "
    "before they choose how to pay."
)


df = pd.read_csv("midterm_project/market_data.csv")

def load_model():
    with open("payment.pkl", "rb") as model_file:
        return pickle.load(model_file)


with st.form("payment_form"):
    left_column, right_column = st.columns(2)

    with left_column:
        city = st.selectbox("City", df["City"].dropna().unique().tolist())
        customer_type = st.selectbox(
            "Customer type", df["Customer type"].dropna().unique().tolist()
        )
        product_line = st.selectbox(
            "Product line", df["Product line"].dropna().unique().tolist()
        )

    with right_column:
        quantity = st.number_input("Quantity", min_value=1, value=1)
        sales = st.number_input(
            "Sales total (including tax)", min_value=0.0, value=20.0, step=0.01
        )

    predict_button = st.form_submit_button("Predict")

if predict_button:
    input_data = pd.DataFrame(
        {
            "City": [city],
            "Customer type": [customer_type],
            "Product line": [product_line],
            "Quantity": [quantity],
            "Sales": [sales],
        }
    )
    model = load_model()
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted payment method: **{prediction}**")

with st.expander("About the prediction"):
    st.write(
        "This app demonstrates a classification model trained on historical supermarket "
        "transactions to predict the most likely payment methid based on the user data "
    )
