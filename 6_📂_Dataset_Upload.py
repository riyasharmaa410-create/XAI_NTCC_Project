import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="XAI - Dataset Upload",
    page_icon="📂",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📂 Dataset Upload & XAI Analysis")

st.markdown(
    """
    Upload a CSV dataset and the application will automatically
    prepare it for Explainable AI analysis.

    **Dataset → Preprocessing → Model Training → Prediction → XAI**

    The application supports both numerical and categorical
    features.
    """
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "data", "txt"],
    help="Upload a CSV, DATA, or TXT file."
)


if uploaded_file is None:

    st.info(
        "Please upload a CSV, DATA, or TXT dataset to begin."
    )

    st.stop()


# ============================================================
# READ DATASET
# ============================================================

st.subheader("📄 Dataset")

header_present = st.checkbox(
    "Does the first row contain column names?",
    value=True
)


try:

    if header_present:

        df = pd.read_csv(uploaded_file)

    else:

        df = pd.read_csv(
            uploaded_file,
            header=None
        )

        df.columns = [
            f"Feature_{i + 1}"
            for i in range(df.shape[1])
        ]

except Exception as e:

    st.error(
        f"Unable to read the uploaded dataset.\n\nError: {e}"
    )

    st.stop()


# ============================================================
# BASIC VALIDATION
# ============================================================

if df.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


if df.shape[1] < 2:

    st.error(
        "The dataset must contain at least two columns."
    )

    st.stop()


# ============================================================
# DATASET PREVIEW
# ============================================================

st.write("### Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.write("### Dataset Information")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)


with info_col1:

    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )


with info_col2:

    st.metric(
        "Columns",
        f"{df.shape[1]:,}"
    )


with info_col3:

    st.metric(
        "Missing Values",
        f"{int(df.isnull().sum().sum()):,}"
    )


with info_col4:

    st.metric(
        "Duplicate Rows",
        f"{int(df.duplicated().sum()):,}"
    )


# ============================================================
# TARGET COLUMN
# ============================================================

st.write("### 🎯 Target Column")

column_list = df.columns.tolist()


# Columns that should never be selected as target
excluded_target_options = [
    "Student_ID",
    "End_of_Semester_Status",
    "Censored"
]


target_options = [
    column
    for column in column_list
    if column not in excluded_target_options
]


if not target_options:

    st.error(
        "No valid target columns are available."
    )

    st.stop()


# Automatically select Target_Dropout_Next_Sem
default_target_index = 0


if "Target_Dropout_Next_Sem" in target_options:

    default_target_index = target_options.index(
        "Target_Dropout_Next_Sem"
    )


target_column = st.selectbox(
    "Choose the target/output column:",
    target_options,
    index=default_target_index
)


# ============================================================
# TARGET INFORMATION
# ============================================================

st.write("### Target Information")

target_data = df[target_column]


target_col1, target_col2, target_col3 = st.columns(3)


with target_col1:

    st.metric(
        "Target Column",
        str(target_column)
    )


with target_col2:

    st.metric(
        "Unique Classes",
        int(target_data.nunique(dropna=True))
    )


with target_col3:

    st.metric(
        "Missing Target Values",
        int(target_data.isnull().sum())
    )


# ============================================================
# TARGET CLASSES
# ============================================================

st.write("### Target Classes")

target_distribution = (
    target_data
    .value_counts(dropna=False)
    .reset_index()
)

target_distribution.columns = [
    "Class",
    "Count"
]

st.dataframe(
    target_distribution,
    use_container_width=True
)


# ============================================================
# REMOVE ROWS WITH MISSING TARGET
# ============================================================

working_df = df.copy()


missing_target_count = int(
    working_df[target_column].isnull().sum()
)


if missing_target_count > 0:

    st.warning(
        f"{missing_target_count} rows contain missing "
        f"target values and will be removed."
    )

    working_df = working_df.dropna(
        subset=[target_column]
    )


working_df = working_df.reset_index(
    drop=True
)


if working_df.empty:

    st.error(
        "No usable rows remain after removing missing target values."
    )

    st.stop()


# ============================================================
# IDENTIFIER DETECTION
# ============================================================

identifier_columns = []


for column in working_df.columns:

    column_lower = str(column).strip().lower()

    if (
        column_lower == "id"
        or column_lower.endswith("_id")
        or column_lower.endswith(" id")
    ):

        if column != target_column:

            identifier_columns.append(column)


# ============================================================
# DATA LEAKAGE PROTECTION
# ============================================================

leakage_columns = []


if target_column == "Target_Dropout_Next_Sem":

    possible_leakage_columns = [
        "End_of_Semester_Status",
        "Censored"
    ]

    for column in possible_leakage_columns:

        if column in working_df.columns:

            leakage_columns.append(column)


# ============================================================
# DISPLAY EXCLUDED COLUMNS
# ============================================================

if identifier_columns:

    st.warning(
        "Identifier columns were excluded from the model features:"
    )

    st.write(
        identifier_columns
    )


if leakage_columns:

    st.warning(
        "Potential outcome-related columns were excluded "
        "from the input features to reduce possible data leakage:"
    )

    st.write(
        leakage_columns
    )

    st.caption(
        "These columns are not used when predicting "
        "Target_Dropout_Next_Sem."
    )


# ============================================================
# CREATE FEATURES AND TARGET
# ============================================================

columns_to_drop = [
    target_column
]


# Remove identifiers

for column in identifier_columns:

    if column not in columns_to_drop:

        columns_to_drop.append(column)


# Remove leakage columns

for column in leakage_columns:

    if column not in columns_to_drop:

        columns_to_drop.append(column)


X = working_df.drop(
    columns=columns_to_drop
).copy()


y = working_df[target_column].copy()


# ============================================================
# CHECK FEATURES
# ============================================================

if X.shape[1] == 0:

    st.error(
        "No input features remain after preprocessing."
    )

    st.stop()


# ============================================================
# REMOVE COMPLETELY EMPTY FEATURES
# ============================================================

empty_feature_columns = [
    column
    for column in X.columns
    if X[column].isnull().all()
]


if empty_feature_columns:

    st.warning(
        "The following columns contain only missing values "
        "and were removed:"
    )

    st.write(
        empty_feature_columns
    )

    X = X.drop(
        columns=empty_feature_columns
    )


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()


categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# ============================================================
# FEATURE INFORMATION
# ============================================================

st.write("### Feature Information")

feature_col1, feature_col2, feature_col3 = st.columns(3)


with feature_col1:

    st.metric(
        "Total Input Features",
        X.shape[1]
    )


with feature_col2:

    st.metric(
        "Numerical Features",
        len(numerical_columns)
    )


with feature_col3:

    st.metric(
        "Categorical Features",
        len(categorical_columns)
    )


# ============================================================
# SHOW NUMERICAL FEATURES
# ============================================================

if numerical_columns:

    with st.expander(
        "🔢 Numerical Features"
    ):

        st.write(
            numerical_columns
        )


# ============================================================
# SHOW CATEGORICAL FEATURES
# ============================================================

if categorical_columns:

    with st.expander(
        "🔤 Categorical Features"
    ):

        st.write(
            categorical_columns
        )


# ============================================================
# NUMERICAL MISSING VALUES
# ============================================================

for column in numerical_columns:

    median_value = X[column].median()

    if pd.isna(median_value):

        median_value = 0

    X[column] = X[column].fillna(
        median_value
    )


# ============================================================
# CATEGORICAL MISSING VALUES
# ============================================================

for column in categorical_columns:

    X[column] = X[column].astype(str)

    X[column] = X[column].replace(
        [
            "nan",
            "None",
            "NaN"
        ],
        "Unknown"
    )

    X[column] = X[column].fillna(
        "Unknown"
    )


# ============================================================
# ONE-HOT ENCODING
# ============================================================

if categorical_columns:

    st.info(
        "Categorical features are automatically converted "
        "into numerical values using one-hot encoding."
    )

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False,
        dtype=int
    )


# ============================================================
# FINAL FEATURE CLEANING
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(0)


for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


X = X.fillna(0)


# ============================================================
# TARGET PREPARATION
# ============================================================

y = y.astype(str)


# ============================================================
# REMOVE RARE CLASSES
# ============================================================

class_counts = y.value_counts()


rare_classes = class_counts[
    class_counts < 2
].index.tolist()


if rare_classes:

    st.warning(
        "The following target classes contain fewer than "
        "2 samples and were removed:"
    )

    st.write(
        rare_classes
    )

    keep_mask = ~y.isin(
        rare_classes
    )

    X = X.loc[
        keep_mask
    ].reset_index(drop=True)

    y = y.loc[
        keep_mask
    ].reset_index(drop=True)


# ============================================================
# TARGET VALIDATION
# ============================================================

if y.nunique() < 2:

    st.error(
        "The selected target must contain at least "
        "two different classes."
    )

    st.stop()


# ============================================================
# PREPROCESSED DATA INFORMATION
# ============================================================

st.write("### ⚙️ Preprocessed Dataset")

prep_col1, prep_col2, prep_col3 = st.columns(3)


with prep_col1:

    st.metric(
        "Samples",
        f"{X.shape[0]:,}"
    )


with prep_col2:

    st.metric(
        "Features After Encoding",
        f"{X.shape[1]:,}"
    )


with prep_col3:

    st.metric(
        "Target Classes",
        f"{y.nunique():,}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

st.write("### 🔀 Train/Test Split")


try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

except ValueError:

    st.warning(
        "Stratified splitting was not possible. "
        "A normal 80/20 split will be used."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


split_col1, split_col2 = st.columns(2)


with split_col1:

    st.metric(
        "Training Samples",
        f"{len(X_train):,}"
    )


with split_col2:

    st.metric(
        "Testing Samples",
        f"{len(X_test):,}"
    )


# ============================================================
# TRAIN MODEL BUTTON
# ============================================================

st.write("### 🌳 Model Training")

st.markdown(
    """
    The Random Forest model will be trained only after you
    click the button below.
    """
)


train_model_button = st.button(
    "🚀 Train Model",
    type="primary",
    use_container_width=True
)


# ============================================================
# TRAIN MODEL
# ============================================================

if train_model_button:

    with st.spinner(
        "Training Random Forest model..."
    ):

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )


    # --------------------------------------------------------
    # SAVE MODEL TO SESSION STATE
    # --------------------------------------------------------

    st.session_state[
        "uploaded_model"
    ] = model


    st.session_state[
        "model_trained"
    ] = True


    st.session_state[
        "uploaded_X_train"
    ] = X_train


    st.session_state[
        "uploaded_X_test"
    ] = X_test


    st.session_state[
        "uploaded_y_train"
    ] = y_train


    st.session_state[
        "uploaded_y_test"
    ] = y_test


    st.session_state[
        "uploaded_feature_names"
    ] = X.columns.tolist()


    st.session_state[
        "uploaded_target_column"
    ] = target_column


    st.session_state[
        "uploaded_dataset"
    ] = working_df


    st.session_state[
        "uploaded_X_original"
    ] = X.copy()


    st.session_state[
        "uploaded_identifier_columns"
    ] = identifier_columns


    st.session_state[
        "uploaded_leakage_columns"
    ] = leakage_columns


    st.session_state[
        "uploaded_classes"
    ] = list(model.classes_)


    st.success(
        "✅ Random Forest model trained successfully."
    )


# ============================================================
# CHECK WHETHER MODEL HAS BEEN TRAINED
# ============================================================

if not st.session_state.get(
    "model_trained",
    False
):

    st.info(
        "👆 Click **🚀 Train Model** to train the Random Forest "
        "model and display the results."
    )

    st.stop()


# ============================================================
# GET TRAINED MODEL
# ============================================================

model = st.session_state[
    "uploaded_model"
]


X_train = st.session_state[
    "uploaded_X_train"
]


X_test = st.session_state[
    "uploaded_X_test"
]


y_train = st.session_state[
    "uploaded_y_train"
]


y_test = st.session_state[
    "uploaded_y_test"
]


# ============================================================
# MODEL PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

correct_predictions = np.sum(
    np.array(y_test) == np.array(y_pred)
)


total_predictions = len(
    y_test
)


accuracy = (
    correct_predictions /
    total_predictions
    if total_predictions > 0
    else 0
)


classes = list(
    model.classes_
)


# ============================================================
# MANUAL PRECISION / RECALL / F1
# ============================================================

precision_values = []
recall_values = []
f1_values = []


for current_class in classes:

    true_positive = np.sum(
        (np.array(y_test) == current_class)
        &
        (np.array(y_pred) == current_class)
    )


    false_positive = np.sum(
        (np.array(y_test) != current_class)
        &
        (np.array(y_pred) == current_class)
    )


    false_negative = np.sum(
        (np.array(y_test) == current_class)
        &
        (np.array(y_pred) != current_class)
    )


    if (
        true_positive + false_positive
    ) > 0:

        precision_class = (
            true_positive /
            (
                true_positive +
                false_positive
            )
        )

    else:

        precision_class = 0


    if (
        true_positive + false_negative
    ) > 0:

        recall_class = (
            true_positive /
            (
                true_positive +
                false_negative
            )
        )

    else:

        recall_class = 0


    if (
        precision_class + recall_class
    ) > 0:

        f1_class = (
            2 *
            precision_class *
            recall_class /
            (
                precision_class +
                recall_class
            )
        )

    else:

        f1_class = 0


    precision_values.append(
        precision_class
    )


    recall_values.append(
        recall_class
    )


    f1_values.append(
        f1_class
    )


precision = (
    np.mean(
        precision_values
    )
    if precision_values
    else 0
)


recall = (
    np.mean(
        recall_values
    )
    if recall_values
    else 0
)


f1 = (
    np.mean(
        f1_values
    )
    if f1_values
    else 0
)


# ============================================================
# SAVE METRICS
# ============================================================

st.session_state[
    "uploaded_accuracy"
] = accuracy


st.session_state[
    "uploaded_precision"
] = precision


st.session_state[
    "uploaded_recall"
] = recall


st.session_state[
    "uploaded_f1"
] = f1


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.write("### 📊 Model Performance")


metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)


with metric_col1:

    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with metric_col2:

    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with metric_col3:

    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with metric_col4:

    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


st.caption(
    "Precision, Recall and F1 Score are macro-averaged "
    "across the target classes."
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.write("### 🔲 Confusion Matrix")


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)


cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual: {str(c)}"
        for c in classes
    ],
    columns=[
        f"Predicted: {str(c)}"
        for c in classes
    ]
)


st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# MODEL DETAILS
# ============================================================

st.write("### 🌳 Model Details")


model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    st.write(
        "**Model:** Random Forest"
    )

    st.write(
        "**Number of Trees:** 200"
    )


with model_col2:

    st.write(
        "**Train/Test Split:** 80/20"
    )

    st.write(
        "**Random State:** 42"
    )


with model_col3:

    st.write(
        f"**Input Features:** {X.shape[1]}"
    )

    st.write(
        f"**Target:** {target_column}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.write("### 🔍 Feature Importance")


try:

    importance_values = model.feature_importances_


    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": importance_values
        }
    )


    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )


    st.dataframe(
        importance_df.head(15),
        use_container_width=True
    )

except Exception:

    st.info(
        "Feature importance could not be displayed."
    )


# ============================================================
# PREDICTION
# ============================================================

st.write("### 🔮 Prediction")


if len(X_test) > 0:

    sample_index = st.number_input(
        "Choose a test sample index:",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )


    selected_sample = X_test.iloc[
        [sample_index]
    ]


    actual_value = y_test.iloc[
        sample_index
    ]


    predicted_value = model.predict(
        selected_sample
    )[0]


    probabilities = model.predict_proba(
        selected_sample
    )[0]


    # --------------------------------------------------------
    # PREDICTION RESULT
    # --------------------------------------------------------

    st.write(
        "#### Prediction Result"
    )


    prediction_col1, prediction_col2, prediction_col3 = st.columns(3)


    with prediction_col1:

        st.metric(
            "Actual Class",
            str(actual_value)
        )


    with prediction_col2:

        st.metric(
            "Predicted Class",
            str(predicted_value)
        )


    with prediction_col3:

        predicted_probability = np.max(
            probabilities
        )


        st.metric(
            "Confidence",
            f"{predicted_probability * 100:.2f}%"
        )


    # --------------------------------------------------------
    # CLASS PROBABILITIES
    # --------------------------------------------------------

    st.write(
        "#### Class Probabilities"
    )


    probability_data = []


    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):

        probability_data.append(
            {
                "Class": str(class_name),
                "Probability": (
                    f"{probability * 100:.2f}%"
                )
            }
        )


    probability_df = pd.DataFrame(
        probability_data
    )


    st.dataframe(
        probability_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    with st.expander(
        "View input features used for this prediction"
    ):

        st.dataframe(
            selected_sample.T,
            use_container_width=True
        )


# ============================================================
# EXPLAINABLE AI SECTION
# ============================================================

st.write("### 💡 Explainable AI")


st.info(
    """
    The uploaded dataset has now been trained with a Random Forest
    model and is ready for Explainable AI analysis.

    The XAI stage will answer:

    **1. What prediction did the model make?**

    **2. Which features influenced that prediction?**

    **3. Which features are most important overall?**

    **4. What happens to the prediction when a feature is changed?**

    SHAP explanations describe model behavior and should not be
    interpreted as proof of causal relationships.
    """
)


# ============================================================
# DATA LEAKAGE INFORMATION
# ============================================================

if target_column == "Target_Dropout_Next_Sem":

    st.write(
        "### ⚠️ Data Leakage Protection"
    )


    st.success(
        """
        For Target_Dropout_Next_Sem, the model does not use:

        • Student_ID  
        • End_of_Semester_Status  
        • Censored  

        as input features.

        This helps prevent information from the outcome or later
        stages of the student record from directly influencing
        the prediction.
        """
    )


# ============================================================
# FINAL STATUS
# ============================================================

st.success(
    "Dataset processing and model training completed successfully."
)


st.caption(
    "Model: Random Forest | Trees: 200 | "
    "Train/Test Split: 80/20 | Random State: 42"
)