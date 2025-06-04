import streamlit as st
import mlflow
import pandas as pd

# Set up MLflow tracking URI (Replace with your actual MLflow instance)
mlflow.set_tracking_uri("https://dagshub.com/VarunRd-124/mlops_project_water_potability.mlflow")

# Define the model name
model_name = "Best Model"

# Initialize Streamlit app
st.title("💧 Water Potability Prediction")
st.write("Enter water quality parameters to predict potability.")

# Fetch latest production model from MLflow
try:
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if versions:
        latest_version = versions[0].version
        run_id = versions[0].run_id
        logged_model = f'runs:/{run_id}/{model_name}'
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        st.success(f"✅ Model loaded (Version: {latest_version})")

        # Create user input fields
        ph = st.number_input("pH Level", value=7.0)
        hardness = st.number_input("Hardness", value=150.0)
        solids = st.number_input("Solids", value=20000.0)
        chloramines = st.number_input("Chloramines", value=7.0)
        sulfate = st.number_input("Sulfate", value=350.0)
        conductivity = st.number_input("Conductivity", value=500.0)
        organic_carbon = st.number_input("Organic Carbon", value=10.0)
        trihalomethanes = st.number_input("Trihalomethanes", value=85.0)
        turbidity = st.number_input("Turbidity", value=3.0)

        # Prediction Button
        if st.button("🔍 Predict Potability"):
            # Prepare input data
            data = pd.DataFrame({
                'ph': [ph],
                'Hardness': [hardness],
                'Solids': [solids],
                'Chloramines': [chloramines],
                'Sulfate': [sulfate],
                'Conductivity': [conductivity],
                'Organic_carbon': [organic_carbon],
                'Trihalomethanes': [trihalomethanes],
                'Turbidity': [turbidity]
            })

            # Generate prediction
            prediction = loaded_model.predict(data)[0]

            # Display result
            st.write("🔹 **Prediction Result:**", "Potable ✅" if prediction == 1 else "Not Potable ❌")
    else:
        st.error("❌ No model found in the 'Production' stage.")

except Exception as e:
    st.error(f"⚠ Error loading model: {e}")