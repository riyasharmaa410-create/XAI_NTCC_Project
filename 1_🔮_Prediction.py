
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

st.title("🔮 Prediction")

st.write(
    """
    Select a test sample and see the prediction made by the active
    machine learning model. When a model has been trained on an
    uploaded dataset, this page uses that model automatically.
    """
)

# ============================================================
# CHECK FOR A TRAINED UPLOADED-DATASET MODEL
# ============================================================

uploaded_results = st.session_state.get("dataset_model")

uploaded_model_available = (
    st.session_state.get("dataset_model_trained", False)
    and uploaded_results is not None
    and isinstance(uploaded_results, dict)
    and "model" in uploaded_results
    and "X_test" in uploaded_results
    and "y_test" in uploaded_results
)

if uploaded_model_available:

    model = uploaded_results["model"]
    X_test = uploaded_results["X_test"]
    y_test = uploaded_results["y_test"]
    feature_names = uploaded_results["feature_names"]
    target_column = uploaded_results["target"]
    task = uploaded_results["task"]

    model_source = "Uploaded Dataset"

else:

    # ========================================================
    # LOAD ORIGINAL MODEL
    # ========================================================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "model")

    MODEL_PATH = os.path.join(
        MODEL_DIR, "xai_random_forest_model.pkl"
    )
    X_TEST_PATH = os.path.join(MODEL_DIR, "X_test.csv")
    Y_TEST_PATH = os.path.join(MODEL_DIR, "y_test.csv")
    FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

    required_files = [
        MODEL_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        FEATURE_PATH,
    ]

    missing_files = [
        path for path in required_files
        if not os.path.isfile(path)
    ]

    if missing_files:
        st.error(
            "No trained uploaded-dataset model is available, "
            "and the original model files are missing."
        )

        st.write("Expected original files:")

        for path in missing_files:
            st.code(os.path.relpath(path, BASE_DIR))

        st.info(
            "Go to Dataset Upload, upload a dataset, select a target, "
            "and click Train Model. Then return to this page."
        )
        st.stop()

    try:
        model = joblib.load(MODEL_PATH)
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()
        feature_names = joblib.load(FEATURE_PATH)

    except Exception as e:
        st.error(f"Could not load the original model: {e}")
        st.stop()

    target_column = "diagnosis"
    task = "Classification"
    model_source = "Original UCI Breast Cancer Model"

# ============================================================
# VALIDATE ACTIVE MODEL DATA
# ============================================================

if len(X_test) == 0:
    st.error("The test dataset contains no samples.")
    st.stop()

if len(y_test) != len(X_test):
    st.error("The test features and target labels have different lengths.")
    st.stop()

if not hasattr(model, "predict"):
    st.error("The active model does not support prediction.")
    st.stop()

if list(X_test.columns) != list(feature_names):
    st.warning(
        "The test feature columns differ from the stored feature names. "
        "Predictions will use the test data's current columns."
    )

# ============================================================
# ACTIVE MODEL INFORMATION
# ============================================================

st.markdown("---")
st.subheader("🤖 Active Model")

info1, info2, info3 = st.columns(3)

info1.metric("Model", type(model).__name__)
info2.metric("Target", str(target_column))
info3.metric("Features", f"{X_test.shape[1]:,}")

st.caption(f"Currently using: **{model_source}**")

if uploaded_model_available:
    st.success(
        "Uploaded dataset model detected. "
        "This page is using the model you trained in Dataset Upload."
    )

# ============================================================
# SELECT TEST SAMPLE
# ============================================================

st.markdown("---")
st.subheader("🔢 Select Test Sample")

sample_index = st.number_input(
    "Test Sample Index",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1,
    key="prediction_sample_index",
)

selected_sample = X_test.iloc[[int(sample_index)]]

# ============================================================
# MAKE PREDICTION
# ============================================================

try:
    prediction = model.predict(selected_sample)[0]
    actual = y_test.iloc[int(sample_index)]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ============================================================
# DISPLAY CLASSIFICATION OR REGRESSION
# ============================================================

st.markdown("---")
st.subheader("🎯 Prediction Result")

if task == "Classification":

    col1, col2 = st.columns(2)
    col1.metric("Actual Class", str(actual))
    col2.metric("Predicted Class", str(prediction))

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(selected_sample)[0]
            model_classes = list(model.classes_)

            probability_df = pd.DataFrame({
                "Class": [str(c) for c in model_classes],
                "Probability (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilities
                ],
            })

            st.markdown("---")
            st.subheader("📈 Class Probabilities")

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                probability_df.set_index("Class")
            )

            st.caption(
                "The highest class probability is not a guarantee "
                "that the prediction is correct."
            )

        except Exception as e:
            st.warning(f"Class probabilities are unavailable: {e}")

else:

    col1, col2 = st.columns(2)
    col1.metric("Actual Value", f"{float(actual):.4f}")
    col2.metric("Predicted Value", f"{float(prediction):.4f}")

    st.caption(
        "This is a regression prediction. Classification probabilities "
        "are not applicable."
    )

# ============================================================
# SHOW INPUT FEATURES
# ============================================================

st.markdown("---")
st.subheader("🔍 Selected Sample Features")

st.dataframe(
    selected_sample.T.rename(columns={selected_sample.index[0]: "Value"}),
    use_container_width=True,
)

# ============================================================
# XAI CONNECTION
# ============================================================

st.markdown("---")
st.subheader("🧠 XAI Connection")

st.info(
    """
    The active model generates the prediction from the selected input
    features. Feature importance can help describe overall model behavior,
    while local explanation methods can help examine an individual
    prediction.

    Feature importance and model explanations do not prove causality
    and do not guarantee that the prediction is correct.
    """
)

# ============================================================
# FINAL STATUS
# ============================================================

st.caption(
    f"Target: {target_column} | Task: {task} | "
    f"Test samples: {len(X_test):,}"
)
