import streamlit as st
import pandas as pd
import numpy as np

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Benzene Concentration Prediction",
    layout="wide"
)

# ===============================
# App Title & Description
# ===============================
st.title("Benzene Concentration Prediction in Isomerization Unit")
st.write(
    "Machine Learning-based analysis and preprocessing of process sensor "
    "and laboratory data for predicting benzene concentration in the "
    "Isomerization Unit feed stream."
)

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader(
    "Upload Process and Lab Data (DCS / LIMS CSV)",
    type=["csv"]
)

# ===============================
# Main Logic
# ===============================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------------
    # Data Preview
    # -------------------------------
    st.subheader("Process and Laboratory Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Dataset Overview
    # -------------------------------
    st.subheader("Operational Data Overview")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    # -------------------------------
    # Data Types
    # -------------------------------
    st.subheader("Sensor & Laboratory Data Types")
    st.write(df.dtypes)

    # -------------------------------
    # Missing Values
    # -------------------------------
    st.subheader("Missing Values by Column")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    # -------------------------------
    # Missing Data Distribution Chart
    # -------------------------------
    st.subheader("Missing Data Distribution")

    total_missing = int(missing_values.sum())
    total_values = df.shape[0] * df.shape[1]
    total_present = total_values - total_missing

    missing_df = pd.DataFrame({
        "Status": ["Available Data", "Missing Data"],
        "Count": [total_present, total_missing]
    })

    st.bar_chart(missing_df.set_index("Status"))

    # -------------------------------
    # Statistical Summary
    # -------------------------------
    st.subheader("Statistical Summary")
    st.write(df.describe())

    # -------------------------------
    # Feature Selection
    # -------------------------------
    st.subheader("Feature Selection")

    all_columns = df.columns.tolist()

    selected_features = st.multiselect(
        "Select features for analysis / modeling",
        options=all_columns,
        default=all_columns
    )

    if selected_features:
        selected_df = df[selected_features]

        st.subheader("Selected Features Preview")
        st.dataframe(selected_df.head())

        st.success(
            f"{len(selected_features)} features selected successfully."
        )
    else:
        st.warning("Please select at least one feature.")

    st.success("Data loaded successfully! Ready for preprocessing!")

else:
    st.info("Please upload a CSV file to begin.")


