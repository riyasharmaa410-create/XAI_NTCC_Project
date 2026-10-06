import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.title("🔮 Prediction")

st.write(
    """
    Select a test sample and see how the trained machine learning
    model classifies it. When an uploaded dataset has been trained,
    this page automatically uses that uploaded model.
    """
)


# ============================================================
# CHECK FOR UPLOADED/TRAINED MODEL
# ============================================================

uploaded_model_available = (
    st.session_state.get("model_trained", False)
    and "uploaded_model" in st.session_state
)


# ============================================================
# USE UPLOADED MODEL IF AVAILABLE
# OTHERWISE USE ORIGINAL XAI MODEL
# ============================================================

if uploaded_model_available:

    # --------------------------------------------------------
    # UPLOADED DATASET MODEL
    # --------------------------------------------------------

    model = st.session_state["uploaded_model"]

    X_test = st.session_state["uploaded_X_test"]
    y_test = st.session_state["uploaded_y_test"]

    feature_names = st.session_state["uploaded_feature_names"]

    target_column = st.session_state[
        "uploaded_target_column"
    ]

    class_names = st.session_state[
        "uploaded_classes"
    ]

    model_source = "Uploaded Dataset"

else:

    # --------------------------------------------------------
    # ORIGINAL PROJECT MODEL
    # --------------------------------------------------------

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "model",
        "xai_random_forest_model.pkl"
    )

    X_TEST_PATH = os.path.join(
        BASE_DIR,
        "model",
        "X_test.csv"
    )

    Y_TEST_PATH = os.path.join(
        BASE_DIR,
        "model",
        "y_test.csv"
    )

    FEATURE_PATH = os.path.join(
        BASE_DIR,
        "model",
        "feature_names.pkl"
    )

    model = joblib.load(
        MODEL_PATH
    )

    X_test = pd.read_csv(
        X_TEST_PATH
    )

    y_test = pd.read_csv(
        Y_TEST_PATH
    ).squeeze()

    feature_names = joblib.load(
        FEATURE_PATH
    )

    class_names = [
        "Benign",
        "Malignant"
    ]

    target_column = "diagnosis"

    model_source = "Original UCI Breast Cancer Model"


# ============================================================
# ACTIVE MODEL INFORMATION
# ============================================================

st.markdown("---")

st.subheader("🤖 Active Model")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Model",
        "Random Forest"
    )

with info2:
    st.metric(
        "Target",
        target_column
    )

with info3:
    st.metric(
        "Features",
        len(feature_names)
    )

st.caption(
    f"Currently using: **{model_source}**"
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
    step=1
)


# ============================================================
# SELECT SAMPLE
# ============================================================

selected_sample = X_test.iloc[
    [int(sample_index)]
]


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    selected_sample
)[0]


probabilities = model.predict_proba(
    selected_sample
)[0]


actual = y_test.iloc[
    int(sample_index)
]


# ============================================================
# CLASS LABEL HANDLING
# ============================================================

model_classes = list(
    model.classes_
)


def get_class_label(class_value):

    # Uploaded dataset classes are encoded as 0, 1, ...
    if uploaded_model_available:

        class_index = model_classes.index(
            class_value
        )

        return str(
            class_names[class_index]
        )

    # Original model
    if int(class_value) == 0:
        return "Benign"

    return "Malignant"


actual_class = get_class_label(
    actual
)

predicted_class = get_class_label(
    prediction
)


# ============================================================
# CONFIDENCE
# ============================================================

confidence = (
    max(probabilities) * 100
)


# ============================================================
# PREDICTION RESULT
# ============================================================

st.markdown("---")

st.subheader("🎯 Prediction Result")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Actual Class",
        actual_class
    )


with col2:

    st.metric(
        "Predicted Class",
        predicted_class
    )


with col3:

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


# ============================================================
# CLASS PROBABILITIES
# ============================================================

st.markdown("---")

st.subheader("📈 Class Probabilities")


probability_data = []


for class_index, class_value in enumerate(
    model_classes
):

    probability_data.append(
        {
            "Class": str(
                class_names[class_index]
            ),
            "Probability": (
                probabilities[class_index]
                * 100
            )
        }
    )


probability_df = pd.DataFrame(
    probability_data
)


st.dataframe(
    probability_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROBABILITY BAR CHART
# ============================================================

st.bar_chart(
    probability_df.set_index("Class")
)


# ============================================================
# SELECTED SAMPLE FEATURES
# ============================================================

st.markdown("---")

st.subheader(
    "🔍 Selected Sample Features"
)

st.dataframe(
    selected_sample,
    use_container_width=True
)


# ============================================================
# XAI CONNECTION
# ============================================================

st.markdown("---")

st.subheader("🧠 XAI Connection")

st.info(
    """
    This prediction is produced by the active Random Forest model.

    The next XAI stage explains **why the model made this prediction**
    by examining the contribution of individual input features using SHAP.

    If an uploaded dataset has been trained, the prediction above
    automatically uses that uploaded dataset's model.
    """
)


# ============================================================
# DATASET STATUS
# ============================================================

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
        **📂 Dataset Upload** page to switch this page
        to the uploaded model.
        """
    )