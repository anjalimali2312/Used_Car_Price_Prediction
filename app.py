
# import streamlit as st
# import pandas as pd
# import pickle
# import os

# # -----------------------------
# # Load Model & Preprocessor
# # -----------------------------

# model_path = os.path.join(
#     "artifacts",
#     "model.pkl"
# )

# preprocessor_path = os.path.join(
#     "artifacts",
#     "preprocessor.pkl"
# )

# with open(model_path, "rb") as file:
#     model = pickle.load(file)

# with open(preprocessor_path, "rb") as file:
#     preprocessor = pickle.load(file)

# # -----------------------------
# # App UI
# # -----------------------------

# st.set_page_config(
#     page_title="Used Car Price Predictor",
#     page_icon="🚗"
# )

# st.title("🚗 Used Car Price Prediction")

# st.write(
#     "Enter vehicle details to estimate its selling price."
# )

# # -----------------------------
# # User Inputs
# # -----------------------------

# year = st.number_input(
#     "Manufacturing Year",
#     min_value=1990,
#     max_value=2024,
#     value=2018
# )

# km_driven = st.number_input(
#     "Kilometers Driven",
#     min_value=0,
#     max_value=1000000,
#     value=30000
# )

# fuel = st.selectbox(
#     "Fuel Type",
#     [
#         "Petrol",
#         "Diesel",
#         "CNG",
#         "LPG",
#         "Electric"
#     ]
# )

# seller_type = st.selectbox(
#     "Seller Type",
#     [
#         "Individual",
#         "Dealer",
#         "Trustmark Dealer"
#     ]
# )

# transmission = st.selectbox(
#     "Transmission",
#     [
#         "Manual",
#         "Automatic"
#     ]
# )

# owner = st.selectbox(
#     "Owner Type",
#     [
#         "First Owner",
#         "Second Owner",
#         "Third Owner",
#         "Fourth & Above Owner",
#         "Test Drive Car"
#     ]
# )

# brand = st.text_input(
#     "Brand",
#     value="Maruti"
# )

# # -----------------------------
# # Prediction
# # -----------------------------

# if st.button("Predict Price"):

#     current_year = 2024

#     car_age = current_year - year

#     input_df = pd.DataFrame({

#         "km_driven": [km_driven],
#         "fuel": [fuel],
#         "seller_type": [seller_type],
#         "transmission": [transmission],
#         "owner": [owner],
#         "Car_Age": [car_age],
#         "Brand": [brand.lower()]

#     })

#     transformed_data = preprocessor.transform(
#         input_df
#     )

#     prediction = model.predict(
#         transformed_data
#     )[0]

#     st.success(
#         f"Estimated Selling Price: ₹ {prediction:,.2f}"
#     )


from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# -----------------------------
# Load Model & Preprocessor
# -----------------------------

model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(preprocessor_path, "rb") as file:
    preprocessor = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    km_driven = float(request.form["km_driven"])
    fuel = request.form["fuel"]
    seller_type = request.form["seller_type"]
    transmission = request.form["transmission"]
    owner = request.form["owner"]
    brand = request.form["brand"]

    current_year = 2024
    car_age = current_year - year

    input_df = pd.DataFrame({
        "km_driven": [km_driven],
        "fuel": [fuel],
        "seller_type": [seller_type],
        "transmission": [transmission],
        "owner": [owner],
        "Car_Age": [car_age],
        "Brand": [brand.lower()]
    })

    transformed_data = preprocessor.transform(input_df)

    prediction = model.predict(transformed_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Estimated Selling Price: ₹ {prediction:,.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)