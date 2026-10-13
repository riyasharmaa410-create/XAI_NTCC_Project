
import streamlit as st
import pandas as pd
import joblib
import os

st.title("🔮 Prediction")

st.write(
    """
    Select a test sample and see how the trained machine learning
    model classifies it. When an uploaded dataset has been trained,
    this page automatically uses that uploaded model.
    """
)

# Check whether an uploaded model is available
uploaded_model_available = (
    st.session_state.get("model_trained", False)
    and "uploaded_model" in st.session_state
)

# Load the active model
if uploaded_model_available:
    model = st.session_state["uploaded_model"]
    X_test = st.session_state["uploaded_X_test"]
    y_test = st.session_state["uploaded_y_test"]
    feature_names = st.session_state["uploaded_feature_names"]
    target_column = st.session_state["uploaded_target_column"]
    class_names = st.session_state["uploaded_classes"]
    model_source = "Uploaded Dataset"

else:
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
        st.error("The original model files are missing from the repository.")
        st.write("The app expects these files inside the `model` folder:")
        for path in missing_files:
            st.code(os.path.relpath(path, BASE_DIR))
        st.info(
            "Please ensure these files exist in the GitHub repository "
            "under the model folder."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()
    feature_names = joblib.load(FEATURE_PATH)

    class_names = ["Benign", "Malignant"]
    target_column = "diagnosis"
    model_source = "Original UCI Breast Cancer Model"

# Validate test data
if len(X_test) == 0:
    st.error("The test dataset contains no samples.")
    st.stop()

if len(y_test) != len(X_test):
    st.error("The test features and labels have different lengths.")
    st.stop()

# Active model information
st.markdown("---")
st.subheader("🤖 Active Model")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Model", type(model).__name__)

with info2:
    st.metric("Target", target_column)

with info3:
    st.metric("Features", len(feature_names))

st.caption(f"Currently using: **{model_source}**")

# Select test sample
st.markdown("---")
st.subheader("🔢 Select Test Sample")

sample_index = st.number_input(
    "Test Sample Index",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1,
)

selected_sample = X_test.iloc[[int(sample_index)]]

# Prediction
prediction = model.predict(selected_sample)[0]
probabilities = model.predict_proba(selected_sample)[0]
actual = y_test.iloc[int(sample_index)]

model_classes = list(model.classes_)

def get_class_label(class_value):
    if uploaded_model_available:
        class_index = model_classes.index(class_value)
        if class_index < len(class_names):
            return str(class_names[class_index])
        return str(class_value)

    if int(class_value) == 0:
        return "Benign"
    return "Malignant"

actual_class = get_class_label(actual)
predicted_class = get_class_label(prediction)
confidence = max(probabilities) * 100

# Prediction result
st.markdown("---")
st.subheader("🎯 Prediction Result")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Actual Class", actual_class)

with col2:
    st.metric("Predicted Class", predicted_class)

with col3:
    st.metric("Confidence", f"{confidence:.2f}%")

# Class probabilities
st.markdown("---")
st.subheader("📈 Class Probabilities")

probability_data = []

for class_index, class_value in enumerate(model_classes):
    if uploaded_model_available and class_index < len(class_names):
        label = str(class_names[class_index])
    elif not uploaded_model_available:
        label = get_class_label(class_value)
    else:
        label = str(class_value)

    probability_data.append({
        "Class": label,
        "Probability (%)": round(
            float(probabilities[class_index]) * 100, 2
        ),
    })

probability_df = pd.DataFrame(probability_data)

st.dataframe(
    probability_df,
    width="stretch",
    hide_index=True,
)

st.bar_chart(
    probability_df.set_index("Class")
)

# Selected sample features
st.markdown("---")
st.subheader("🔍 Selected Sample Features")

st.dataframe(
    selected_sample,
    width="stretch",
)

# XAI connection
st.markdown("---")
st.subheader("🧠 XAI Connection")

st.info(
    """
    This prediction is produced by the active machine learning model.

    The next XAI stage explains why the model made this prediction
    by examining the contribution of individual input features using SHAP.

    If an uploaded dataset has been trained, this page uses that
    uploaded dataset's model.
    """
)

# Dataset status
if uploaded_model_available:
    st.success(
        f"""
        **Uploaded dataset model is active.**

        Target: `{target_column}`

        Features: `{len(feature_names)}`

        Test samples: `{len(X_test)}`
        """
    )
else:
    st.info(
        """
        **Original project model is active.**

        Upload and train a dataset from the
        **📂 Dataset Upload** page to switch to the uploaded model.
        """
    )
