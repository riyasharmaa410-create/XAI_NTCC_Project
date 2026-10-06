import streamlit as st
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# CHECK ACTIVE MODEL
# ==========================================

if not st.session_state.get("model_trained", False):

    st.warning(
        "No uploaded dataset model is currently active."
    )

    st.info(
        "Please go to the 📂 Dataset Upload page, "
        "upload your dataset, select the target column, "
        "and train the model first."
    )

    st.stop()


# ==========================================
# LOAD ACTIVE MODEL
# ==========================================

model = st.session_state["uploaded_model"]

X_test = st.session_state["uploaded_X_test"]

y_test = st.session_state["uploaded_y_test"]

feature_names = st.session_state["uploaded_feature_names"]

target_column = st.session_state.get(
    "uploaded_target_column",
    "Target"
)


# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

class_labels = list(model.classes_)


# ==========================================
# PAGE TITLE
# ==========================================

st.title("📊 Model Performance")

st.write(
    """
    This page shows the performance of the currently
    trained Random Forest model on the held-out test
    dataset.
    """
)

st.markdown("---")


# ==========================================
# ACTIVE MODEL INFORMATION
# ==========================================

st.header("🤖 Active Model")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:

    st.metric(
        "Model",
        "Random Forest"
    )

with info_col2:

    st.metric(
        "Features",
        len(feature_names)
    )

with info_col3:

    st.metric(
        "Test Samples",
        len(X_test)
    )

with info_col4:

    st.metric(
        "Trees",
        200
    )

st.write(
    f"**Target Column:** `{target_column}`"
)


# ==========================================
# PERFORMANCE METRICS
# ==========================================

st.markdown("---")

st.header("📈 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    st.metric(
        "Macro Precision",
        f"{precision * 100:.2f}%"
    )


with col3:

    st.metric(
        "Macro Recall",
        f"{recall * 100:.2f}%"
    )


with col4:

    st.metric(
        "Macro F1 Score",
        f"{f1 * 100:.2f}%"
    )


# ==========================================
# METRIC EXPLANATION
# ==========================================

st.markdown("---")

st.subheader("📖 Understanding the Metrics")

st.write(
    """
    **Accuracy:** Percentage of all test samples that
    were classified correctly.

    **Macro Precision:** Precision is calculated separately
    for each class and then averaged, giving equal importance
    to each class.

    **Macro Recall:** Recall is calculated separately for
    each class and then averaged, giving equal importance
    to each class.

    **Macro F1 Score:** Harmonic mean of precision and recall
    calculated across classes and then averaged.
    """
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

st.markdown("---")

st.header("🔢 Confusion Matrix")

cm_labels = [
    f"Actual {label}"
    for label in class_labels
]

cm_columns = [
    f"Predicted {label}"
    for label in class_labels
]

cm_df = pd.DataFrame(
    cm,
    index=cm_labels,
    columns=cm_columns
)

st.dataframe(
    cm_df,
    width="stretch"
)


# ==========================================
# CLASS DISTRIBUTION
# ==========================================

st.markdown("---")

st.header("📋 Test Dataset Class Distribution")

actual_distribution = (
    y_test.value_counts()
    .sort_index()
)

predicted_distribution = (
    pd.Series(predictions)
    .value_counts()
    .sort_index()
)

distribution_df = pd.DataFrame(
    {
        "Class": [
            str(label)
            for label in class_labels
        ],
        "Actual Samples": [
            int(actual_distribution.get(label, 0))
            for label in class_labels
        ],
        "Predicted Samples": [
            int(predicted_distribution.get(label, 0))
            for label in class_labels
        ]
    }
)

st.dataframe(
    distribution_df,
    width="stretch"
)


# ==========================================
# MODEL CONFIGURATION
# ==========================================

st.markdown("---")

st.header("⚙️ Model Configuration")

configuration_df = pd.DataFrame(
    {
        "Parameter": [
            "Model",
            "Number of Trees",
            "Training/Test Split",
            "Random State",
            "Number of Features",
            "Test Samples",
            "Target Column"
        ],
        "Value": [
            "Random Forest Classifier",
            "200",
            "80% / 20%",
            "42",
            str(len(feature_names)),
            str(len(X_test)),
            target_column
        ]
    }
)

st.dataframe(
    configuration_df,
    width="stretch"
)


# ==========================================
# INTERPRETATION
# ==========================================

st.markdown("---")

st.header("💡 Performance Interpretation")

st.write(
    f"""
    The trained Random Forest model achieved an accuracy
    of **{accuracy * 100:.2f}%** on the held-out test dataset.

    The macro-averaged precision is
    **{precision * 100:.2f}%**, the macro-averaged recall is
    **{recall * 100:.2f}%**, and the macro-averaged F1 score is
    **{f1 * 100:.2f}%**.

    These metrics describe how the model performed on this
    particular test dataset. They should not be interpreted
    as a guarantee of performance on every future dataset.
    """
)


# ==========================================
# XAI CONNECTION
# ==========================================

st.markdown("---")

st.info(
    """
    Model performance and Explainable AI serve different
    purposes.

    Performance metrics show **how well the model predicts**.

    XAI techniques show **how the model arrived at individual
    predictions**.

    Together, they provide information about both model
    performance and model behaviour.
    """
)


# ==========================================
# TECHNICAL NOTE
# ==========================================

with st.expander("🔧 Technical Details"):

    st.write(
        f"""
        **Algorithm:** Random Forest Classifier

        **Number of trees:** 200

        **Train/Test split:** 80% / 20%

        **Random state:** 42

        **Input features:** {len(feature_names)}

        **Test samples:** {len(X_test)}

        **Target:** {target_column}

        **Evaluation:** Predictions are generated on the
        held-out test dataset stored in the active session.
        """
    )


# ==========================================
# DISCLAIMER
# ==========================================

st.markdown("---")

st.caption(
    "Note: Performance metrics are calculated from the "
    "currently trained uploaded-dataset model and its "
    "held-out test set. Results may differ for other "
    "datasets or future data."
)