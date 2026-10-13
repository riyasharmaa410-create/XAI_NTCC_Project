```python
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="What-If Analysis | XAI",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 What-If Analysis")
st.write(
    "Change the value of a feature and observe how the "
    "machine learning model's prediction changes."
)

st.divider()

# --------------------------------------------------
# 1. GET THE TRAINED DATASET MODEL
# --------------------------------------------------

dataset_model = st.session_state.get("dataset_model")
model_trained = st.session_state.get(
    "dataset_model_trained", False
)

if not model_trained or not isinstance(dataset_model, dict):
    st.warning(
        "No uploaded dataset model is currently active."
    )
    st.info(
        "Go to the 📂 Dataset Upload page, upload your "
        "dataset, select the target column, and train "
        "the model first."
    )
    st.stop()

model = dataset_model.get("model")
task = dataset_model.get("task")
feature_names = dataset_model.get("feature_names", [])
X_train = dataset_model.get("X_train")

if model is None or X_train is None or not feature_names:
    st.error(
        "The trained model is missing required information. "
        "Please train the model again on the Dataset Upload page."
    )
    st.stop()

# Ensure the training features are in DataFrame format.
if isinstance(X_train, pd.DataFrame):
    training_data = X_train.copy()
else:
    training_data = pd.DataFrame(
        X_train, columns=list(feature_names)
    )

# Match the model's feature order.
feature_names = list(feature_names)

if len(feature_names) != training_data.shape[1]:
    st.error(
        "The feature names do not match the training data. "
        "Please train the model again."
    )
    st.stop()

training_data = training_data[feature_names]

# --------------------------------------------------
# 2. SHOW MODEL INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Task", str(task).title())

with col2:
    st.metric("Input Features", len(feature_names))

with col3:
    st.metric("Training Samples", len(training_data))

st.divider()

# --------------------------------------------------
# 3. CREATE FEATURE INPUTS
# --------------------------------------------------

st.subheader("🎛️ Change Feature Values")
st.caption(
    "The initial values are taken from a typical training "
    "sample. Adjust the controls and click Analyze."
)

# Use the median training row as a reasonable starting point.
initial_row = training_data.median(numeric_only=True)

# Fill any non-numeric feature defaults using the first row.
if not training_data.empty:
    first_row = training_data.iloc[0]

    for feature in feature_names:
        if feature not in initial_row.index:
            initial_row.loc[feature] = first_row[feature]

# Keep the input controls in a form so changes are analyzed
# only when the user clicks the button.
with st.form("what_if_form"):
    edited_values = {}
    columns = st.columns(2)

    for index, feature in enumerate(feature_names):
        with columns[index % 2]:
            series = training_data[feature]
            numeric_series = pd.to_numeric(
                series, errors="coerce"
            )
            is_numeric = numeric_series.notna().all()

            if is_numeric:
                minimum = float(numeric_series.min())
                maximum = float(numeric_series.max())
                default_value = float(
                    numeric_series.median()
                )

                if not np.isfinite(minimum) or not np.isfinite(maximum):
                    edited_values[feature] = default_value
                elif minimum == maximum:
                    edited_values[feature] = st.number_input(
                        feature,
                        value=default_value,
                        key=f"whatif_{feature}",
                        help="This feature has a constant training value."
                    )
                else:
                    edited_values[feature] = st.slider(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=min(
                            max(default_value, minimum), maximum
                        ),
                        key=f"whatif_{feature}"
                    )
            else:
                choices = series.dropna().unique().tolist()

                if not choices:
                    st.warning(
                        f"No valid training values for {feature}."
                    )
                    edited_values[feature] = None
                else:
                    edited_values[feature] = st.selectbox(
                        feature,
                        options=choices,
                        key=f"whatif_{feature}"
                    )

    analyze = st.form_submit_button(
        "🔍 Analyze Prediction",
        use_container_width=True
    )

# --------------------------------------------------
# 4. MAKE THE PREDICTION
# --------------------------------------------------

if analyze:
    input_df = pd.DataFrame(
        [edited_values],
        columns=feature_names
    )

    st.divider()
    st.subheader("📊 Analysis Results")

    try:
        prediction = model.predict(input_df)[0]

        if task == "classification":
            classes = getattr(model, "classes_", None)

            # Most sklearn classifiers expose predict_proba.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]

                probability_df = pd.DataFrame({
                    "Class": [
                        str(c) for c in classes
                    ] if classes is not None else [
                        f"Class {i}" for i in range(len(probabilities))
                    ],
                    "Probability": probabilities
                })

                probability_df["Probability (%)"] = (
                    probability_df["Probability"] * 100
                ).round(2)

                left, right = st.columns([1, 1])

                with left:
                    st.markdown("#### Predicted Class")
                    st.success(str(prediction))
                    st.caption(
                        "This is the model's prediction, not a "
                        "guarantee about the real-world outcome."
                    )

                with right:
                    st.markdown("#### Class Probabilities")
                    st.dataframe(
                        probability_df[
                            ["Class", "Probability (%)"]
                        ],
                        use_container_width=True,
                        hide_index=True
                    )

                st.markdown("#### Probability Chart")
                chart_df = probability_df.set_index("Class")[
                    "Probability (%)"
                ]
                st.bar_chart(chart_df)

                # Show how confident the model is in the predicted class.
                if classes is not None:
                    class_list = list(classes)
                    if prediction in class_list:
                        pred_index = class_list.index(prediction)
                        confidence = float(
                            probabilities[pred_index] * 100
                        )
                        st.metric(
                            "Probability for Predicted Class",
                            f"{confidence:.2f}%"
                        )

            else:
                st.markdown("#### Predicted Class")
                st.success(str(prediction))
                st.info(
                    "This model does not support predict_proba(), "
                    "so class probabilities cannot be displayed."
                )

        else:
            prediction_value = float(prediction)
            st.metric(
                "Predicted Value",
                f"{prediction_value:.4f}"
            )

            st.caption(
                "This is the model's predicted numeric value."
            )

        # --------------------------------------------------
        # 5. COMPARE INPUT WITH THE TYPICAL TRAINING VALUES
        # --------------------------------------------------

        st.divider()
        st.subheader("🔎 Feature Comparison")

        comparison_rows = []

        for feature in feature_names:
            try:
                baseline = float(
                    pd.to_numeric(
                        training_data[feature],
                        errors="coerce"
                    ).median()
                )
                current = float(edited_values[feature])

                comparison_rows.append({
                    "Feature": feature,
                    "Training Median": baseline,
                    "Your Value": current,
                    "Difference": current - baseline
                })
            except (TypeError, ValueError):
                comparison_rows.append({
                    "Feature": feature,
                    "Training Median": "Categorical",
                    "Your Value": str(edited_values[feature]),
                    "Difference": "Not applicable"
                })

        comparison_df = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # 6. MODEL FEATURE IMPORTANCE, IF AVAILABLE
        # --------------------------------------------------

        if hasattr(model, "feature_importances_"):
            st.divider()
            st.subheader("🌟 Global Feature Importance")

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values(
                "Importance", ascending=False
            )

            st.bar_chart(
                importance_df.head(10).set_index("Feature")[
                    "Importance"
                ]
            )

            st.caption(
                "These values describe overall model importance. "
                "They are not a causal explanation of this individual prediction."
            )

        elif hasattr(model, "coef_"):
            st.divider()
            st.subheader("🌟 Model Coefficients")

            coefficients = np.asarray(model.coef_)

            if coefficients.ndim == 1:
                coef_values = coefficients
            else:
                # For multiclass linear models, show the first class's coefficients.
                coef_values = coefficients[0]

            if len(coef_values) == len(feature_names):
                coefficient_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Coefficient": coef_values
                })
                coefficient_df["Absolute Coefficient"] = (
                    coefficient_df["Coefficient"].abs()
                )
                coefficient_df = coefficient_df.sort_values(
                    "Absolute Coefficient", ascending=False
                )

                st.bar_chart(
                    coefficient_df.head(10).set_index("Feature")[
                        "Coefficient"
                    ]
                )

                st.caption(
                    "Coefficient direction and magnitude depend on "
                    "the model and feature scaling."
                )

    except Exception as error:
        st.error(
            "The model could not analyze these inputs. "
            "Check that the Dataset Upload page used the same "
            "feature columns and preprocessing."
        )
        st.code(str(error))

# --------------------------------------------------
# 7. EXPLANATION NOTE
# --------------------------------------------------

st.divider()
st.subheader("💡 How to Interpret What-If Analysis")

st.write(
    """
    - Change one or more feature values.
    - Click **Analyze Prediction** to run the model again.
    - Compare the new predicted class, probability, or numeric value.
    - Repeat with different values to observe how the output changes.

    **Important:** A changed prediction shows how this model responds
    to the supplied inputs. It does not prove that changing a feature
    causes a real-world outcome, and it does not by itself establish
    why the model made its original prediction.
    """
)
```
