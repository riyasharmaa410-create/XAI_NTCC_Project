import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Feature Explanation",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🔍 XAI Feature Explanation")

st.write(
    """
    This page explains **why the active machine learning model made
    a particular prediction**.

    The explanation is generated using a **model-based local feature
    contribution method**. Each feature is changed individually while
    the other features remain unchanged, and the resulting change in
    prediction probability is measured.
    """
)

st.info(
    """
    **Technical note:** This page does not use SHAP because the current
    Windows environment is blocking a required Numba DLL.

    The contribution values shown here are therefore **model-based
    feature contributions, not SHAP values**.
    """
)


# ============================================================
# CHECK WHETHER MODEL HAS BEEN TRAINED
# ============================================================

if not st.session_state.get("model_trained", False):

    st.warning(
        """
        No trained model is currently available.

        Please go to **📂 Dataset Upload**, upload a dataset, select
        a target column, and train the Random Forest model first.
        """
    )

    st.stop()


# ============================================================
# LOAD ACTIVE MODEL AND DATA
# ============================================================

model = st.session_state.get(
    "uploaded_model"
)

X_test = st.session_state.get(
    "uploaded_X_test"
)

y_test = st.session_state.get(
    "uploaded_y_test"
)

feature_names = st.session_state.get(
    "uploaded_feature_names",
    []
)

target_column = st.session_state.get(
    "uploaded_target_column",
    "Target"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

if model is None:
    st.error("The trained model could not be loaded.")
    st.stop()


if X_test is None:
    st.error("The test dataset could not be loaded.")
    st.stop()


if y_test is None:
    st.error("The test target values could not be loaded.")
    st.stop()


if len(feature_names) == 0:
    feature_names = list(X_test.columns)


# Convert feature names to strings for safe display
feature_names = [
    str(feature)
    for feature in feature_names
]


# ============================================================
# ALIGN TEST DATA WITH FEATURE NAMES
# ============================================================

try:

    X_test = X_test.copy()

    # Ensure the feature order matches the trained model
    if list(X_test.columns) != list(feature_names):

        if all(
            feature in X_test.columns
            for feature in feature_names
        ):

            X_test = X_test[
                feature_names
            ]

        else:

            st.error(
                """
                The uploaded test dataset and the trained model
                do not contain the same feature columns.
                """
            )

            st.stop()

except Exception as e:

    st.error(
        f"Could not prepare the test dataset: {e}"
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("📊 Current XAI Model")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:

    st.metric(
        "Model",
        "Random Forest"
    )

with info_col2:

    st.metric(
        "Target",
        str(target_column)
    )

with info_col3:

    st.metric(
        "Features",
        len(feature_names)
    )

with info_col4:

    st.metric(
        "Test Samples",
        len(X_test)
    )

st.success(
    "Currently using: **Uploaded Dataset Model**"
)


# ============================================================
# SELECT TEST SAMPLE
# ============================================================

st.subheader("🔮 Select a Test Sample")

if len(X_test) == 0:

    st.error(
        "The test dataset contains no samples."
    )

    st.stop()


sample_number = st.number_input(
    "Test sample number:",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1
)


sample_index = int(
    sample_number
)


selected_sample = X_test.iloc[
    [sample_index]
].copy()


# ============================================================
# PREDICTION
# ============================================================

try:

    prediction = model.predict(
        selected_sample
    )[0]

except Exception as e:

    st.error(
        f"Could not generate prediction: {e}"
    )

    st.stop()


# ============================================================
# PREDICTION PROBABILITIES
# ============================================================

try:

    probabilities = model.predict_proba(
        selected_sample
    )[0]

    classes = list(
        model.classes_
    )

except Exception as e:

    st.error(
        f"Could not calculate prediction probabilities: {e}"
    )

    st.stop()


# ============================================================
# PREDICTION CLASS INDEX
# ============================================================

try:

    predicted_class_index = classes.index(
        prediction
    )

except ValueError:

    predicted_class_index = int(
        np.argmax(probabilities)
    )


prediction_probability = float(
    probabilities[predicted_class_index]
)


# ============================================================
# ACTUAL CLASS
# ============================================================

try:

    actual_class = y_test.iloc[
        sample_index
    ]

except Exception:

    actual_class = "Unknown"


# ============================================================
# DISPLAY PREDICTION
# ============================================================

st.subheader("🎯 Prediction")

prediction_col1, prediction_col2, prediction_col3 = st.columns(3)

with prediction_col1:

    st.metric(
        "Actual Class",
        str(actual_class)
    )

with prediction_col2:

    st.metric(
        "Predicted Class",
        str(prediction)
    )

with prediction_col3:

    st.metric(
        "Confidence",
        f"{prediction_probability * 100:.2f}%"
    )


# ============================================================
# CLASS PROBABILITIES
# ============================================================

st.subheader("📈 Class Probabilities")

probability_table = pd.DataFrame(
    {
        "Class": [
            str(value)
            for value in classes
        ],
        "Probability": [
            round(
                float(value) * 100,
                2
            )
            for value in probabilities
        ]
    }
)

st.bar_chart(
    probability_table.set_index(
        "Class"
    )
)


# ============================================================
# SAFE VALUE CONVERSION FUNCTION
# ============================================================

def get_safe_reference_value(
    series
):
    """
    Returns a reference value that is compatible with the
    original feature dtype.

    Handles:
    - binary features
    - integer features
    - floating-point features
    - boolean features
    - categorical/object features
    - unusual/mixed values
    """

    clean_series = series.dropna()

    # ---------------------------------------------
    # Empty column
    # ---------------------------------------------

    if len(clean_series) == 0:

        if pd.api.types.is_bool_dtype(series):

            return False

        if pd.api.types.is_integer_dtype(series):

            return 0

        if pd.api.types.is_float_dtype(series):

            return 0.0

        return ""


    # ---------------------------------------------
    # Boolean
    # ---------------------------------------------

    if pd.api.types.is_bool_dtype(series):

        mode_values = clean_series.mode()

        if len(mode_values) > 0:

            return bool(
                mode_values.iloc[0]
            )

        return bool(
            clean_series.iloc[0]
        )


    # ---------------------------------------------
    # Binary numerical / one-hot feature
    # ---------------------------------------------

    try:

        unique_values = set(
            clean_series.unique()
        )

        if unique_values.issubset(
            {0, 1}
        ):

            mode_values = clean_series.mode()

            if len(mode_values) > 0:

                reference = mode_values.iloc[0]

            else:

                reference = clean_series.iloc[0]

            if pd.api.types.is_integer_dtype(
                series
            ):

                return int(
                    reference
                )

            return reference

    except Exception:

        pass


    # ---------------------------------------------
    # Integer feature
    # ---------------------------------------------

    if pd.api.types.is_integer_dtype(
        series
    ):

        try:

            median_value = clean_series.median()

            return int(
                round(
                    float(median_value)
                )
            )

        except Exception:

            try:

                return int(
                    clean_series.iloc[0]
                )

            except Exception:

                return 0


    # ---------------------------------------------
    # Floating-point feature
    # ---------------------------------------------

    if pd.api.types.is_float_dtype(
        series
    ):

        try:

            return float(
                clean_series.median()
            )

        except Exception:

            try:

                return float(
                    clean_series.iloc[0]
                )

            except Exception:

                return 0.0


    # ---------------------------------------------
    # Categorical / object / string feature
    # ---------------------------------------------

    if (
        pd.api.types.is_object_dtype(series)
        or
        pd.api.types.is_categorical_dtype(series)
    ):

        try:

            mode_values = clean_series.mode()

            if len(mode_values) > 0:

                return mode_values.iloc[0]

            return clean_series.iloc[0]

        except Exception:

            return clean_series.iloc[0]


    # ---------------------------------------------
    # General fallback
    # ---------------------------------------------

    try:

        return clean_series.median()

    except Exception:

        return clean_series.iloc[0]


# ============================================================
# SAFE FEATURE ASSIGNMENT
# ============================================================

def safely_assign_value(
    dataframe,
    row_index,
    column,
    value
):
    """
    Safely assigns a reference value while respecting
    the dtype expected by the model.
    """

    try:

        column_dtype = dataframe[column].dtype

        # -----------------------------------------
        # Boolean
        # -----------------------------------------

        if pd.api.types.is_bool_dtype(
            column_dtype
        ):

            dataframe.loc[
                row_index,
                column
            ] = bool(value)

            return dataframe


        # -----------------------------------------
        # Integer
        # -----------------------------------------

        if pd.api.types.is_integer_dtype(
            column_dtype
        ):

            dataframe.loc[
                row_index,
                column
            ] = int(
                round(
                    float(value)
                )
            )

            return dataframe


        # -----------------------------------------
        # Float
        # -----------------------------------------

        if pd.api.types.is_float_dtype(
            column_dtype
        ):

            dataframe.loc[
                row_index,
                column
            ] = float(value)

            return dataframe


        # -----------------------------------------
        # Other types
        # -----------------------------------------

        dataframe.loc[
            row_index,
            column
        ] = value

        return dataframe


    except Exception:

        # Last-resort dtype conversion for this
        # individual column.

        try:

            dataframe[column] = dataframe[
                column
            ].astype(object)

            dataframe.loc[
                row_index,
                column
            ] = value

        except Exception:

            pass

        return dataframe


# ============================================================
# MODEL-BASED FEATURE CONTRIBUTION
# ============================================================

st.subheader("🧠 Model Feature Contribution")

st.write(
    """
    Each feature is changed individually while the other input
    features remain unchanged.

    The change in the predicted probability of the selected
    prediction class is used as the feature's local contribution.
    """
)

st.markdown(
    """
    **Positive contribution:** the original feature value supports
    the selected prediction.

    **Negative contribution:** the original feature value moves the
    model away from the selected prediction.

    This explains **model behavior**, not real-world causation.
    """
)


# ============================================================
# CALCULATE FEATURE CONTRIBUTIONS
# ============================================================

contribution_results = []

progress_bar = st.progress(
    0
)

status_text = st.empty()


number_of_features = len(
    feature_names
)


for feature_number, feature in enumerate(
    feature_names
):

    status_text.write(
        f"Analyzing feature {feature_number + 1} "
        f"of {number_of_features}: **{feature}**"
    )

    try:

        # -----------------------------------------
        # Original probability
        # -----------------------------------------

        original_probability = float(
            probabilities[
                predicted_class_index
            ]
        )


        # -----------------------------------------
        # Create modified sample
        # -----------------------------------------

        modified_sample = selected_sample.copy()


        # -----------------------------------------
        # Find safe reference value
        # -----------------------------------------

        reference_value = get_safe_reference_value(
            X_test[feature]
        )


        # -----------------------------------------
        # Assign reference value safely
        # -----------------------------------------

        modified_sample = safely_assign_value(
            modified_sample,
            modified_sample.index[0],
            feature,
            reference_value
        )


        # -----------------------------------------
        # New prediction probability
        # -----------------------------------------

        modified_probabilities = model.predict_proba(
            modified_sample
        )[0]


        modified_probability = float(
            modified_probabilities[
                predicted_class_index
            ]
        )


        # -----------------------------------------
        # Contribution
        # -----------------------------------------

        contribution = (
            original_probability
            -
            modified_probability
        )


        contribution_results.append(
            {
                "Feature": feature,
                "Original Value": selected_sample.iloc[
                    0
                ][feature],
                "Reference Value": reference_value,
                "Original Probability": (
                    original_probability * 100
                ),
                "Modified Probability": (
                    modified_probability * 100
                ),
                "Contribution": (
                    contribution * 100
                ),
                "Absolute Contribution": (
                    abs(contribution) * 100
                )
            }
        )


    except Exception as e:

        # -----------------------------------------
        # If one feature fails, continue with
        # remaining features.
        # -----------------------------------------

        contribution_results.append(
            {
                "Feature": feature,
                "Original Value": selected_sample.iloc[
                    0
                ][feature],
                "Reference Value": "Unavailable",
                "Original Probability": (
                    prediction_probability * 100
                ),
                "Modified Probability": np.nan,
                "Contribution": 0.0,
                "Absolute Contribution": 0.0
            }
        )


    progress_bar.progress(
        int(
            (
                (feature_number + 1)
                /
                number_of_features
            )
            * 100
        )
    )


status_text.empty()
progress_bar.empty()


# ============================================================
# CONTRIBUTION DATAFRAME
# ============================================================

contribution_df = pd.DataFrame(
    contribution_results
)


if contribution_df.empty:

    st.error(
        "No feature contributions could be calculated."
    )

    st.stop()


# Remove invalid rows from visualization calculations
valid_contributions = contribution_df[
    np.isfinite(
        contribution_df[
            "Absolute Contribution"
        ]
    )
].copy()


# ============================================================
# SORT BY IMPORTANCE
# ============================================================

top_features = valid_contributions.sort_values(
    by="Absolute Contribution",
    ascending=False
).head(10)


# ============================================================
# MOST INFLUENTIAL FEATURES
# ============================================================

st.subheader(
    "⭐ Most Influential Features"
)

if len(top_features) > 0:

    chart_data = top_features[
        [
            "Feature",
            "Absolute Contribution"
        ]
    ].copy()

    chart_data = chart_data.set_index(
        "Feature"
    )

    st.bar_chart(
        chart_data
    )

else:

    st.info(
        "No feature contribution values are available."
    )


# ============================================================
# CONTRIBUTION TABLE
# ============================================================

st.subheader(
    "📋 Feature Contribution Details"
)

display_table = top_features[
    [
        "Feature",
        "Original Value",
        "Reference Value",
        "Contribution"
    ]
].copy()

display_table[
    "Contribution"
] = display_table[
    "Contribution"
].round(4)

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# POSITIVE CONTRIBUTIONS
# ============================================================

st.subheader(
    "➕ Positive Contributions"
)

positive_features = valid_contributions[
    valid_contributions[
        "Contribution"
    ] > 0
].sort_values(
    by="Contribution",
    ascending=False
).head(10)


if len(positive_features) > 0:

    positive_table = positive_features[
        [
            "Feature",
            "Contribution"
        ]
    ].copy()

    positive_table[
        "Contribution"
    ] = positive_table[
        "Contribution"
    ].round(4)

    st.dataframe(
        positive_table,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No positive feature contributions were found."
    )


# ============================================================
# NEGATIVE CONTRIBUTIONS
# ============================================================

st.subheader(
    "➖ Negative Contributions"
)

negative_features = valid_contributions[
    valid_contributions[
        "Contribution"
    ] < 0
].sort_values(
    by="Contribution"
).head(10)


if len(negative_features) > 0:

    negative_table = negative_features[
        [
            "Feature",
            "Contribution"
        ]
    ].copy()

    negative_table[
        "Contribution"
    ] = negative_table[
        "Contribution"
    ].round(4)

    st.dataframe(
        negative_table,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No negative feature contributions were found."
    )


# ============================================================
# FEATURE CONTRIBUTION IMPORTANCE
# ============================================================

st.subheader(
    "📊 Feature Contribution Importance"
)

importance_data = valid_contributions[
    [
        "Feature",
        "Absolute Contribution"
    ]
].sort_values(
    by="Absolute Contribution",
    ascending=False
).head(15)

if len(importance_data) > 0:

    importance_chart = importance_data.set_index(
        "Feature"
    )

    st.bar_chart(
        importance_chart
    )


# ============================================================
# PREDICTION CONTRIBUTION
# ============================================================

st.subheader(
    "📈 Prediction Contribution"
)

prediction_contribution_data = valid_contributions[
    [
        "Feature",
        "Contribution"
    ]
].sort_values(
    by="Contribution",
    ascending=False
).head(15)

if len(prediction_contribution_data) > 0:

    prediction_chart = (
        prediction_contribution_data
        .set_index("Feature")
    )

    st.bar_chart(
        prediction_chart
    )


# ============================================================
# EXPLANATION
# ============================================================

st.subheader(
    "💡 How to Read This Explanation"
)

if len(top_features) > 0:

    strongest_feature = top_features.iloc[0]

    strongest_feature_name = (
        strongest_feature["Feature"]
    )

    strongest_contribution = float(
        strongest_feature[
            "Contribution"
        ]
    )

    if strongest_contribution > 0:

        st.success(
            f"""
            The feature with the largest absolute local contribution
            is **{strongest_feature_name}**.

            Its contribution is approximately
            **{strongest_contribution:.2f} percentage points**
            toward the selected prediction class.
            """
        )

    elif strongest_contribution < 0:

        st.warning(
            f"""
            The feature with the largest absolute local contribution
            is **{strongest_feature_name}**.

            Its contribution is approximately
            **{strongest_contribution:.2f} percentage points**,
            meaning its original value moves the model away from the
            selected prediction class compared with the reference value.
            """
        )

    else:

        st.info(
            f"""
            The largest calculated contribution belongs to
            **{strongest_feature_name}**, but its measured contribution
            is approximately zero for this prediction.
            """
        )


st.write(
    """
    The explanation compares the model's original prediction with
    predictions obtained after replacing one feature at a time with
    a representative reference value.

    Therefore, the contribution describes how the **trained model**
    responds to that controlled feature change.

    It should not be interpreted as proof that the feature causes the
    real-world outcome.
    """
)


# ============================================================
# COMPLETE SELECTED SAMPLE
# ============================================================

with st.expander(
    "📋 View Complete Selected Sample"
):

    sample_display = selected_sample.T.reset_index()

    sample_display.columns = [
        "Feature",
        "Value"
    ]

    st.dataframe(
        sample_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PIPELINE
# ============================================================

st.subheader(
    "🔄 XAI Pipeline"
)

pipeline_col1, pipeline_col2, pipeline_col3, pipeline_col4 = (
    st.columns(4)
)

with pipeline_col1:

    st.markdown(
        """
        ### 📂 Dataset

        Uploaded dataset is prepared
        and encoded.
        """
    )

with pipeline_col2:

    st.markdown(
        """
        ### 🌳 Model

        Random Forest learns
        patterns in the data.
        """
    )

with pipeline_col3:

    st.markdown(
        """
        ### 🔮 Prediction

        The model generates
        a class prediction.
        """
    )

with pipeline_col4:

    st.markdown(
        """
        ### 🔍 Explanation

        Individual feature changes
        are used to study model behavior.
        """
    )


# ============================================================
# TECHNICAL NOTE
# ============================================================

with st.expander(
    "🔧 Technical Details"
):

    st.write(
        """
        **Explanation method**

        A model-based local feature contribution approach is used.

        For each feature:

        1. The original prediction probability is recorded.
        2. That feature is replaced with a representative reference
           value.
        3. The model makes another prediction.
        4. The difference between the two probabilities is recorded
           as the local feature contribution.

        Binary and one-hot encoded features use their most frequent
        value as the reference.

        Integer features use a rounded median.

        Continuous numerical features use their median.

        This implementation is designed to avoid invalid values when
        working with integer encoded features.
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    """
    ⚠️ Disclaimer: These explanations describe the behavior of the
    trained machine learning model. They do not establish causal
    relationships and should not be treated as independent real-world
    decisions.
    """
)