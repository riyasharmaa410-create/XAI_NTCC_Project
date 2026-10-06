import streamlit as st
import pandas as pd


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.title("🧪 What-If Analysis")

st.markdown(
    "Change the value of a feature and observe how the "
    "machine learning model's prediction probability changes."
)

st.markdown("---")


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
# LOAD ACTIVE MODEL FROM SESSION STATE
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
# MODEL INFORMATION
# ==========================================

st.success("Using the currently trained uploaded-dataset model.")

info_col1, info_col2, info_col3 = st.columns(3)

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


st.caption(
    f"Target column: **{target_column}**"
)

st.markdown("---")


# ==========================================
# SELECT TEST SAMPLE
# ==========================================

st.subheader("🔢 Select Test Sample")

sample_index = st.number_input(
    "Select test sample index",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)

selected_sample = X_test.iloc[[sample_index]].copy()

actual_class = y_test.iloc[sample_index]


# ==========================================
# ORIGINAL PREDICTION
# ==========================================

original_prediction = model.predict(
    selected_sample
)[0]

original_probabilities = model.predict_proba(
    selected_sample
)[0]

class_labels = list(model.classes_)

original_prediction_index = list(
    model.classes_
).index(original_prediction)

original_prediction_probability = (
    original_probabilities[original_prediction_index] * 100
)


# ==========================================
# SELECT FEATURE
# ==========================================

st.subheader("🎛️ Select Feature to Modify")

selected_feature = st.selectbox(
    "Choose a feature:",
    feature_names
)


# ==========================================
# ORIGINAL VALUE
# ==========================================

original_value = selected_sample[
    selected_feature
].iloc[0]


try:
    original_value = float(original_value)
except (TypeError, ValueError):
    st.error(
        "The selected feature is not numeric and "
        "cannot be modified using this What-If analysis."
    )
    st.stop()


# ==========================================
# MODIFIED VALUE
# ==========================================

modified_value = st.number_input(
    f"Modified value for {selected_feature}",
    value=float(original_value),
    format="%.4f"
)


st.markdown("---")


# ==========================================
# ORIGINAL FEATURE VALUE
# ==========================================

st.subheader("📌 Original Feature Value")

st.write(
    f"Original value of **{selected_feature}**: "
    f"**{original_value:.4f}**"
)


# ==========================================
# CREATE MODIFIED SAMPLE
# ==========================================

modified_sample = selected_sample.copy()

modified_sample.loc[
    modified_sample.index[0],
    selected_feature
] = modified_value


# ==========================================
# MODIFIED PREDICTION
# ==========================================

modified_prediction = model.predict(
    modified_sample
)[0]

modified_probabilities = model.predict_proba(
    modified_sample
)[0]

modified_prediction_index = list(
    model.classes_
).index(modified_prediction)

modified_prediction_probability = (
    modified_probabilities[modified_prediction_index] * 100
)


# ==========================================
# CALCULATE PROBABILITY CHANGE
# ==========================================

probability_change = (
    modified_prediction_probability
    - original_prediction_probability
)


# ==========================================
# CLASS NAMES
# ==========================================

def format_class_name(value):
    """
    Convert class value into a readable label.
    """

    return str(value)


original_class_name = format_class_name(
    original_prediction
)

modified_class_name = format_class_name(
    modified_prediction
)

actual_class_name = format_class_name(
    actual_class
)


# ==========================================
# PREDICTION SUMMARY
# ==========================================

st.markdown("---")

st.subheader("🔮 Prediction Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:

    st.write("**Actual Class**")

    st.write(
        f"### {actual_class_name}"
    )

with summary_col2:

    st.write("**Original Prediction**")

    st.write(
        f"### {original_class_name}"
    )

with summary_col3:

    st.write("**Modified Prediction**")

    st.write(
        f"### {modified_class_name}"
    )


# ==========================================
# PROBABILITY COMPARISON
# ==========================================

st.markdown("---")

st.subheader("📊 Prediction Probability Comparison")

comparison_col1, comparison_col2 = st.columns(2)


with comparison_col1:

    st.markdown("### Original")

    st.metric(
        "Prediction Probability",
        f"{original_prediction_probability:.2f}%"
    )

    st.write(
        f"Prediction: **{original_class_name}**"
    )


with comparison_col2:

    st.markdown("### Modified")

    st.metric(
        "Prediction Probability",
        f"{modified_prediction_probability:.2f}%"
    )

    st.write(
        f"Prediction: **{modified_class_name}**"
    )


# ==========================================
# PROBABILITY CHANGE
# ==========================================

st.markdown("---")

st.subheader("📈 Change in Prediction Probability")

st.metric(
    "Change",
    f"{probability_change:+.2f} percentage points"
)


# ==========================================
# ALL CLASS PROBABILITIES
# ==========================================

st.markdown("---")

st.subheader("📋 Class Probability Details")

probability_data = pd.DataFrame(
    {
        "Class": [
            format_class_name(class_value)
            for class_value in class_labels
        ],
        "Original Probability (%)": [
            probability * 100
            for probability in original_probabilities
        ],
        "Modified Probability (%)": [
            probability * 100
            for probability in modified_probabilities
        ]
    }
)

probability_data["Change (percentage points)"] = (
    probability_data["Modified Probability (%)"]
    - probability_data["Original Probability (%)"]
)

st.dataframe(
    probability_data,
    width="stretch"
)


# ==========================================
# FEATURE COMPARISON
# ==========================================

st.markdown("---")

st.subheader("🔎 Feature Change")

feature_comparison = pd.DataFrame(
    {
        "Value": [
            original_value,
            modified_value
        ]
    },
    index=[
        "Original",
        "Modified"
    ]
)

st.dataframe(
    feature_comparison,
    width="stretch"
)


# ==========================================
# COMPLETE SAMPLE COMPARISON
# ==========================================

with st.expander(
    "📋 View Complete Original and Modified Sample"
):

    sample_comparison = pd.DataFrame(
        {
            "Original": selected_sample.iloc[0],
            "Modified": modified_sample.iloc[0]
        }
    )

    sample_comparison["Change"] = (
        sample_comparison["Modified"]
        - sample_comparison["Original"]
    )

    st.dataframe(
        sample_comparison,
        width="stretch"
    )


# ==========================================
# EXPLANATION
# ==========================================

st.markdown("---")

st.subheader("💡 How to Interpret This")

st.write(
    f"""
    The What-If analysis changes only the selected feature
    **{selected_feature}** while keeping the other input
    features unchanged.

    The model then makes a new prediction using the modified
    input.

    The difference between the original and modified
    prediction probabilities shows how the trained model
    responds to this controlled change.
    """
)


# ==========================================
# XAI CONNECTION
# ==========================================

st.info(
    "This is an XAI technique for exploring model behaviour. "
    "It shows how changing an input value can affect the "
    "model's prediction. The result should not be interpreted "
    "as proof that the feature causes the real-world outcome."
)


# ==========================================
# TECHNICAL NOTE
# ==========================================

with st.expander("🔧 Technical Details"):

    st.write(
        f"""
        **Model:** Random Forest

        **Target:** {target_column}

        **Number of input features:** {len(feature_names)}

        **Selected test sample:** {sample_index}

        **Modified feature:** {selected_feature}

        **Original value:** {original_value:.4f}

        **Modified value:** {modified_value:.4f}

        The model prediction is calculated twice:
        once using the original sample and once using the
        modified sample.

        Only the selected feature is changed during the
        What-If experiment.
        """
    )


# ==========================================
# DISCLAIMER
# ==========================================

st.markdown("---")

st.caption(
    "Note: What-If analysis demonstrates model behaviour "
    "under controlled input changes. It does not establish "
    "causation and should not be treated as a real-world "
    "decision by itself."
)