import pickle
import pandas as pd
import streamlit as st

st.title("Supermarket Payment Prediction - Hein Ko Midterm Project")

st.sidebar.image("Secondary Logo_vertical-01.png")
st.sidebar.markdown("Name - Hein Ko Ko Oo")
st.sidebar.markdown("Student ID - PIUS20240075")
st.sidebar.markdown("Class - CS300-Fall 26")

st.write(
    "Enter the checkout details to estimate the customer's payment method "
    "before they choose how to pay."
)


df = pd.read_csv("market_data.csv")

def load_model():
    with open("payment.pkl", "rb") as model_file:
        model = pickle.load(model_file)
        try:
            evaluation = pickle.load(model_file)
        except EOFError:
            # Older pickle files contain only the trained model.
            evaluation = None
    return model, evaluation


model, evaluation = load_model()


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
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted payment method: **{prediction}**")

with st.expander("Model evaluation"):
        st.write(f"Evaluated on {evaluation['test_rows']} held-out invoices.")
        st.write(f"Test accuracy: {evaluation['accuracy']:.1%}")
        st.write(f"Macro precision: {evaluation['macro_precision']:.1%}")
