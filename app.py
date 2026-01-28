import streamlit as st
import pandas as pd
import numpy as np

# ===============================
# SAFE ML IMPORTS
# ===============================
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    sklearn_available = True
except Exception:
    sklearn_available = False

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Benzene Concentration Prediction",
    layout="wide"
)

st.title("Benzene Concentration Prediction in Isomerization Unit")
st.write(
    "Interactive feature selection, correlation analysis, and machine learning "
    "for industrial process data."
)

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload Process / Lab Data (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ===============================
# DATA PREVIEW
# ===============================
st.subheader("Data Preview")
st.dataframe(df.head())

c1, c2 = st.columns(2)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])

# ===============================
# FEATURE SELECTION (USER CONTROLLED – KEY FIX)
# ===============================
st.subheader("Feature Selection")

all_columns = df.columns.tolist()

features = st.multiselect(
    "Select input features (you decide what is numeric)",
    all_columns
)

target = st.selectbox(
    "Select target variable (Benzene concentration)",
    all_columns
)

if not features or not target:
    st.warning("Select at least one feature and a target.")
    st.stop()

if target in features:
    features.remove(target)

# ===============================
# SAFE NUMERIC CONVERSION (ONLY SELECTED COLUMNS)
# ===============================
working_df = df[features + [target]].copy()

for col in working_df.columns:
    working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

# Drop rows where target is missing
working_df = working_df.dropna(subset=[target])

# ===============================
# FEATURE COMPARISON (CORRECT)
# ===============================
if len(features) >= 2:
    st.subheader("Feature Comparison")

    fx = st.selectbox("Feature X", features)
    fy = st.selectbox("Feature Y", features, index=1)

    comp_df = working_df[[fx, fy]].dropna()

    if len(comp_df) > 2:
        st.scatter_chart(comp_df, x=fx, y=fy)
        st.line_chart(comp_df)

        corr = comp_df[fx].corr(comp_df[fy])

        st.metric("Pearson Correlation", round(corr, 4))

        if abs(corr) >= 0.8:
            st.success("Strong correlation detected")
        elif abs(corr) >= 0.5:
            st.warning("Moderate correlation detected")
        else:
            st.info("Weak correlation detected")

        st.download_button(
            "Download Comparison Data",
            comp_df.to_csv(index=False),
            file_name=f"{fx}_vs_{fy}.csv"
        )
    else:
        st.warning("Not enough valid data points for correlation.")

# ===============================
# MACHINE LEARNING
# ===============================
st.subheader("Machine Learning")

if not sklearn_available:
    st.error(
        "Machine learning libraries are unavailable in this environment.\n"
        "EDA and correlation work correctly.\n"
        "Run ML locally if required."
    )
    st.stop()

X = working_df[features]
y = working_df[target]

if len(X) < 5:
    st.error("Not enough data for model training.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

if st.button("Train Random Forest Model"):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)

    m1, m2, m3 = st.columns(3)
    m1.metric("R²", round(r2, 4))
    m2.metric("RMSE", round(rmse, 4))
    m3.metric("MAE", round(mae, 4))

    # ===============================
    # FEATURE IMPORTANCE
    # ===============================
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(importance_df.set_index("Feature"))

    # ===============================
    # FINAL PREDICTION
    # ===============================
    st.subheader("Final Benzene Concentration Prediction")

    input_vals = {}
    for f in features:
        input_vals[f] = st.number_input(
            f"Enter value for {f}",
            float(X[f].mean())
        )

    if st.button("Predict"):
        pred = model.predict(pd.DataFrame([input_vals]))[0]
        st.success(
            f"Predicted Benzene Concentration: **{round(pred, 4)}**"
        )
