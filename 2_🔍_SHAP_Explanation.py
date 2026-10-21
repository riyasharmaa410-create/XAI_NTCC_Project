
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="XAI Feature Explanation",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 XAI Feature Explanation")

st.markdown(
    """
    Understand **why a machine learning model made a prediction**.

    This page changes one feature at a time to a reference value and observes
    how the prediction changes. This helps identify which features influence
    the model's output for the selected sample.

    **Important:** These are perturbation-based feature contributions, not
    SHAP values. They explain the behaviour of the trained model and do not
    establish medical, scientific, or causal conclusions.
    """
)

st.divider()

# ============================================================
# 1. CHECK AND LOAD THE CURRENTLY TRAINED MODEL
# ============================================================

uploaded_results = st.session_state.get("dataset_model")

required_keys = [
    "model",
    "task",
    "target",
    "feature_names",
    "X_test",
    "y_test",
    "y_pred",
]

model_available = (
    st.session_state.get("dataset_model_trained", False)
    and isinstance(uploaded_results, dict)
    and all(key in uploaded_results for key in required_keys)
)

if not model_available:
    st.warning(
        """
        **No trained uploaded-dataset model is available.**

        1. Open **Dataset Upload**.
        2. Upload a CSV or Excel dataset.
        3. Select the target column.
        4. Click **Train Model**.
        5. Return to this page.
        """
    )
    st.stop()

model = uploaded_results["model"]

task = str(uploaded_results["task"]).strip().lower()

target_column = uploaded_results["target"]
feature_names = list(uploaded_results["feature_names"])
X_test = uploaded_results["X_test"].copy()
y_test = uploaded_results["y_test"]
y_pred = uploaded_results["y_pred"]

if X_test.empty:
    st.error("The test dataset is empty. Please train the model again.")
    st.stop()

if len(feature_names) == 0:
    st.error("No input features were found. Please check your dataset.")
    st.stop()

# Ensure the test data has the same feature order as the trained model.
missing_features = [
    feature for feature in feature_names
    if feature not in X_test.columns
]

if missing_features:
    st.error(
        "Some trained features are missing from the test data: "
        + ", ".join(missing_features)
    )
    st.stop()

X_test = X_test[feature_names].copy()

# ============================================================
# 2. MODEL AND DATASET INFORMATION
# ============================================================

st.subheader("📊 Current Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Task Type", str(task).title())

with col2:
    st.metric("Target Column", str(target_column))

with col3:
    st.metric("Input Features", len(feature_names))

with col4:
    st.metric("Test Samples", len(X_test))

st.caption(
    "This page uses the model currently trained on the Dataset Upload page."
)

st.divider()

# ============================================================
# 3. SELECT A TEST SAMPLE
# ============================================================

st.subheader("🧪 Select a Sample to Explain")

sample_number = st.selectbox(
    "Choose a test sample",
    options=list(range(len(X_test))),
    format_func=lambda i: f"Test sample {i + 1}",
)

sample = X_test.iloc[[sample_number]].copy()

# ============================================================
# 4. GET THE MODEL'S ORIGINAL PREDICTION
# ============================================================

try:
    original_prediction = model.predict(sample)[0]
except Exception as error:
    st.error(f"Could not generate the prediction: {error}")
    st.stop()

actual_value = y_test.iloc[sample_number] if hasattr(
    y_test, "iloc"
) else y_test[sample_number]

st.subheader("🎯 Model Prediction")

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.write("**Predicted value**")
    st.success(str(original_prediction))

with pred_col2:
    st.write("**Actual value**")
    st.info(str(actual_value))

if task == "classification" and hasattr(model, "predict_proba"):
    try:
        probabilities = model.predict_proba(sample)[0]
        classes = model.classes_

        probability_df = pd.DataFrame({
            "Class": [str(c) for c in classes],
            "Probability (%)": probabilities * 100,
        })

        st.write("**Prediction probabilities**")
        st.dataframe(
            probability_df.style.format({
                "Probability (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig_prob, ax_prob = plt.subplots(figsize=(8, 3.5))
        ax_prob.bar(
            probability_df["Class"],
            probability_df["Probability (%)"],
        )
        ax_prob.set_ylabel("Probability (%)")
        ax_prob.set_xlabel("Class")
        ax_prob.set_ylim(0, 100)
        ax_prob.set_title("Model confidence across classes")
        plt.tight_layout()
        st.pyplot(fig_prob)
        plt.close(fig_prob)

    except Exception as error:
        st.warning(
            "Class probabilities could not be displayed: "
            f"{error}"
        )

elif task == "regression":
    st.caption(
        "This is a regression prediction. The model predicts a numerical "
        "value rather than a class probability."
    )

st.divider()

# ============================================================
# 5. REFERENCE VALUES FOR FEATURES
# ============================================================

def get_reference_value(series):
    """
    Choose a representative reference value for a feature.
    Numerical features use the median; categorical features use the mode.
    """
    non_missing = series.dropna()

    if non_missing.empty:
        return np.nan

    if pd.api.types.is_bool_dtype(series):
        modes = non_missing.mode()
        return modes.iloc[0] if not modes.empty else non_missing.iloc[0]

    if pd.api.types.is_numeric_dtype(series):
        return non_missing.median()

    modes = non_missing.mode()
    return modes.iloc[0] if not modes.empty else non_missing.iloc[0]


def safely_assign_value(dataframe, column, value):
    """
    Replace one feature value in a copy of the selected sample.
    """
    changed = dataframe.copy()

    try:
        changed.loc[changed.index[0], column] = value
    except (TypeError, ValueError):
        changed[column] = changed[column].astype("object")
        changed.loc[changed.index[0], column] = value

    return changed


# ============================================================
# 6. CALCULATE PERTURBATION-BASED CONTRIBUTIONS
# ============================================================

st.subheader("🔬 How Each Feature Influences the Prediction")

st.markdown(
    """
    For each feature, this method replaces its selected-sample value with
    a representative value from the test dataset, keeping the other features
    unchanged. It then measures how the model output changes.

    - **Positive contribution:** The change is in the direction of the
      selected prediction, according to the method used below.
    - **Negative contribution:** The change is away from the selected
      prediction.
    - **Near zero:** Changing that feature alone had little effect on the
      measured output.
    """
)

with st.spinner("Calculating feature contributions..."):

    reference_values = {
        feature: get_reference_value(X_test[feature])
        for feature in feature_names
    }

    contributions = []
    prediction_changes = []

    try:
        if task == "classification":
            if not hasattr(model, "predict_proba"):
                st.error(
                    "This classification model does not support "
                    "predict_proba, so probability-based explanations "
                    "cannot be calculated."
                )
                st.stop()

            classes = list(model.classes_)
            predicted_class = original_prediction

            if predicted_class not in classes:
                st.error("The predicted class was not found in model.classes_.")
                st.stop()

            predicted_class_index = classes.index(predicted_class)
            original_output = float(
                model.predict_proba(sample)[0][predicted_class_index]
            )

            for feature in feature_names:
                changed_sample = safely_assign_value(
                    sample,
                    feature,
                    reference_values[feature],
                )

                changed_output = float(
                    model.predict_proba(changed_sample)[0][
                        predicted_class_index
                    ]
                )

                # Positive means the original feature value raises
                # the probability of the class predicted for this sample.
                contribution = original_output - changed_output

                contributions.append({
                    "Feature": feature,
                    "Original Value": sample.iloc[0][feature],
                    "Reference Value": reference_values[feature],
                    "Contribution": contribution * 100,
                    "Changed Output": changed_output * 100,
                })

                prediction_changes.append({
                    "Feature": feature,
                    "Original Output": original_output * 100,
                    "Changed Output": changed_output * 100,
                    "Change": (changed_output - original_output) * 100,
                })

            output_label = (
                f"Probability of class {predicted_class}"
            )
            output_unit = "percentage points"

        elif task == "regression":
            original_output = float(
                np.asarray(model.predict(sample)).reshape(-1)[0]
            )

            for feature in feature_names:
                changed_sample = safely_assign_value(
                    sample,
                    feature,
                    reference_values[feature],
                )

                changed_output = float(
                    np.asarray(
                        model.predict(changed_sample)
                    ).reshape(-1)[0]
                )

                # Positive means the original feature value raises
                # the prediction relative to its reference replacement.
                contribution = original_output - changed_output

                contributions.append({
                    "Feature": feature,
                    "Original Value": sample.iloc[0][feature],
                    "Reference Value": reference_values[feature],
                    "Contribution": contribution,
                    "Changed Output": changed_output,
                })

                prediction_changes.append({
                    "Feature": feature,
                    "Original Output": original_output,
                    "Changed Output": changed_output,
                    "Change": changed_output - original_output,
                })

            output_label = "Predicted value"
            output_unit = "target units"

        else:
            st.error(
                f"Unsupported task type: {task}. "
                "Choose classification or regression on Dataset Upload."
            )
            st.stop()

    except Exception as error:
        st.error(
            "An error occurred while calculating feature contributions. "
            "Check that the model was trained with the current dataset "
            f"and feature columns.\n\nDetails: {error}"
        )
        st.stop()

contribution_df = pd.DataFrame(contributions)

if contribution_df.empty:
    st.warning("No feature contributions could be calculated.")
    st.stop()

contribution_df["Absolute Contribution"] = (
    contribution_df["Contribution"].abs()
)

contribution_df = contribution_df.sort_values(
    "Absolute Contribution",
    ascending=False,
).reset_index(drop=True)

# ============================================================
# 7. DISPLAY THE MOST IMPORTANT FEATURES
# ============================================================

st.subheader("🏆 Most Influential Features")

top_n = st.slider(
    "Number of features to display",
    min_value=3,
    max_value=min(20, len(feature_names)),
    value=min(10, len(feature_names)),
)

top_features = contribution_df.head(top_n).copy()

st.caption(
    f"Contributions are measured in {output_unit}. "
    "They are local, one-feature-at-a-time changes, not global feature "
    "importance scores."
)

display_df = top_features[
    [
        "Feature",
        "Original Value",
        "Reference Value",
        "Contribution",
        "Changed Output",
    ]
].copy()

if task == "classification":
    display_df["Contribution"] = display_df["Contribution"].map(
        lambda value: f"{value:+.3f} pp"
    )
    display_df["Changed Output"] = display_df["Changed Output"].map(
        lambda value: f"{value:.2f}%"
    )
else:
    display_df["Contribution"] = display_df["Contribution"].map(
        lambda value: f"{value:+.6g}"
    )
    display_df["Changed Output"] = display_df["Changed Output"].map(
        lambda value: f"{value:.6g}"
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 8. VISUALIZE FEATURE CONTRIBUTIONS
# ============================================================

st.subheader("📈 Feature Contribution Chart")

plot_df = top_features.sort_values("Contribution", ascending=True)

fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))

bars = ax.barh(
    plot_df["Feature"].astype(str),
    plot_df["Contribution"],
)

ax.axvline(0, linewidth=1)
ax.set_xlabel(f"Contribution ({output_unit})")
ax.set_ylabel("Feature")
ax.set_title("Local feature contributions for the selected sample")

for bar, value in zip(bars, plot_df["Contribution"]):
    if np.isfinite(value):
        offset = 0.02 * max(
            1.0,
            float(plot_df["Contribution"].abs().max()),
        )
        x_pos = value + offset if value >= 0 else value - offset
        alignment = "left" if value >= 0 else "right"

        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3g}",
            va="center",
            ha=alignment,
            fontsize=8,
        )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

# ============================================================
# 9. AUTOMATIC EXPLANATION
# ============================================================

st.subheader("📝 Explanation in Simple Language")

strongest = contribution_df.iloc[0]
strongest_feature = strongest["Feature"]
strongest_contribution = float(strongest["Contribution"])
strongest_original = strongest["Original Value"]
strongest_reference = strongest["Reference Value"]

st.markdown(
    f"""
    For **test sample {sample_number + 1}**, the model predicted
    **{original_prediction}** for target column **{target_column}**.

    The feature with the largest absolute one-feature-at-a-time effect was
    **{strongest_feature}**.

    - Value in this sample: **{strongest_original}**
    - Reference value: **{strongest_reference}**
    - Measured contribution: **{strongest_contribution:+.4g} {output_unit}**
    """
)

if task == "classification":
    if strongest_contribution > 0:
        st.success(
            f"Replacing **{strongest_feature}** with its reference value "
            f"reduced the probability of the predicted class by "
            f"{strongest_contribution:.3f} percentage points. "
            "The original value therefore increased that class probability "
            "relative to this particular reference replacement."
        )
    elif strongest_contribution < 0:
        st.info(
            f"Replacing **{strongest_feature}** with its reference value "
            f"increased the probability of the predicted class by "
            f"{abs(strongest_contribution):.3f} percentage points. "
            "The original value therefore reduced that class probability "
            "relative to this particular reference replacement."
        )
    else:
        st.info(
            f"Replacing **{strongest_feature}** with its reference value "
            "did not change the measured probability to the displayed "
            "precision."
        )

else:
    if strongest_contribution > 0:
        st.success(
            f"The original value of **{strongest_feature}** produced a "
            f"prediction {strongest_contribution:.4g} target units higher "
            "than the prediction obtained after replacing it with the "
            "reference value."
        )
    elif strongest_contribution < 0:
        st.info(
            f"The original value of **{strongest_feature}** produced a "
            f"prediction {abs(strongest_contribution):.4g} target units "
            "lower than the prediction obtained after replacing it with "
            "the reference value."
        )
    else:
        st.info(
            f"Replacing **{strongest_feature}** with its reference value "
            "did not change the prediction to the displayed precision."
        )

# ============================================================
# 10. SAMPLE DETAILS
# ============================================================

with st.expander("🔎 View the selected sample's feature values"):
    sample_display = sample.T.reset_index()
    sample_display.columns = ["Feature", "Value"]

    st.dataframe(
        sample_display,
        use_container_width=True,
        hide_index=True,
    )

with st.expander("🧰 View all calculated contributions"):
    all_contributions = contribution_df[
        [
            "Feature",
            "Original Value",
            "Reference Value",
            "Contribution",
            "Changed Output",
            "Absolute Contribution",
        ]
    ].copy()

    st.dataframe(
        all_contributions,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# 11. METHODOLOGY AND LIMITATIONS
# ============================================================

st.divider()
st.subheader("ℹ️ How this explanation works")

st.markdown(
    """
    **Method used: local feature perturbation**

    1. The trained model predicts an output for the selected test sample.
    2. A reference value is chosen for each feature using the test dataset:
       the median for numerical columns and the mode for categorical columns.
    3. One feature is replaced by its reference value while the other
       feature values are kept unchanged.
    4. The model predicts again.
    5. The difference between the original output and the changed output
       is reported as that feature's local contribution.

    **Limitations**

    - This method is not SHAP or another additive feature-attribution method.
    - The result depends on the chosen reference values.
    - Features may be correlated, so changing one feature independently can
      create combinations that are uncommon in real data.
    - A small one-feature effect does not prove that a feature is unimportant
      in other samples.
    - These explanations describe model behaviour, not causation.
    - For classification, contributions are changes in the probability of
      the class originally predicted.
    - For regression, contributions are changes in the numerical prediction.
    """
)

st.info(
    "For a new dataset or a changed target column, retrain the model on "
    "Dataset Upload before using these explanations."
)

