import streamlit as st
import pandas as pd
import numpy as np
 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
 
# ============================================================
# PAGE CONFIGURATION
# ============================================================
 
st.set_page_config(
    page_title="XAI - Dataset Upload",
    page_icon="📂",
    layout="wide",
)
 
st.title("📂 Dataset Upload & XAI Analysis")
 
st.markdown(
    """
    Upload a dataset, select the column you want to predict,
    and train a model automatically.
 
    **Upload → Select Target → Preprocess → Train → Evaluate → Explain**
    """
)
 
# ============================================================
# SESSION STATE
# ============================================================
 
if "dataset_model_trained" not in st.session_state:
    st.session_state.dataset_model_trained = False
 
# ============================================================
# FILE UPLOAD
# ============================================================
 
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "data", "txt"],
    help="Upload a CSV, DATA, or TXT file.",
)
 
if uploaded_file is None:
    st.info("Upload a dataset to begin.")
    st.stop()
 
header_present = st.checkbox(
    "Does the first row contain column names?",
    value=True,
)
 
# ============================================================
# READ DATASET
# ============================================================
 
try:
    uploaded_file.seek(0)
 
    df = pd.read_csv(
        uploaded_file,
        header=0 if header_present else None,
        sep=None,
        engine="python",
    )
 
    if not header_present:
        df.columns = [
            f"Feature_{i + 1}" for i in range(df.shape[1])
        ]
 
    # Make duplicate column names unique.
    original_columns = [str(c).strip() for c in df.columns]
    seen = {}
    unique_columns = []
 
    for name in original_columns:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique_columns.append(
            name if count == 0 else f"{name}_{count}"
        )
 
    df.columns = unique_columns
 
except Exception as e:
    st.error(f"Could not read this dataset: {e}")
    st.stop()
 
# ============================================================
# BASIC VALIDATION
# ============================================================
 
if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()
 
if df.shape[1] < 2:
    st.error("The dataset must have at least two columns.")
    st.stop()
 
# Avoid collisions in Streamlit widget identity when the same
# filename is uploaded with different contents.
dataset_signature = (
    uploaded_file.name,
    uploaded_file.size,
    tuple(df.columns),
    len(df),
)
 
if st.session_state.get("dataset_signature") != dataset_signature:
    st.session_state.dataset_signature = dataset_signature
    st.session_state.dataset_model_trained = False
    st.session_state.pop("dataset_model", None)
 
# ============================================================
# DATASET PREVIEW AND INFORMATION
# ============================================================
 
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
 
st.subheader("📊 Dataset Information")
 
c1, c2, c3, c4 = st.columns(4)
 
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
 
# ============================================================
# TARGET SELECTION
# ============================================================
 
st.subheader("🎯 Select Your Target Column")
 
st.write(
    "Choose the column that you want the model to predict. "
    "All other eligible columns will be considered input features."
)
 
target_options = list(df.columns)
 
target_column = st.selectbox(
    "Target/output column",
    options=target_options,
    index=None,
    placeholder="Choose a column to predict...",
    key="dataset_target_selector",
)
 
if target_column is None:
    st.info("Select a target column to continue.")
    st.stop()
 
# Reset the trained model whenever the target changes.
previous_target = st.session_state.get("dataset_last_target")
 
if previous_target != target_column:
    st.session_state.dataset_model_trained = False
    st.session_state.pop("dataset_model", None)
    st.session_state.dataset_last_target = target_column
 
# ============================================================
# DETERMINE TASK TYPE
# ============================================================
 
raw_target = df[target_column]
target_non_missing = raw_target.dropna()
 
if target_non_missing.empty:
    st.error("The selected target contains no usable values.")
    st.stop()
 
# Numeric targets with many distinct values are usually regression.
# Low-cardinality numeric targets are treated as classification.
numeric_target = pd.to_numeric(
    target_non_missing, errors="coerce"
)
 
all_numeric = numeric_target.notna().all()
unique_count = target_non_missing.nunique()
 
if all_numeric:
    numeric_unique_count = numeric_target.nunique()
    is_integer_target = np.allclose(
        numeric_target.to_numpy(dtype=float) % 1, 0
    )
 
    if (
        is_integer_target
        and numeric_unique_count <= 20
    ):
        inferred_task = "Classification"
    elif numeric_unique_count <= 2:
        inferred_task = "Classification"
    else:
        inferred_task = "Regression"
else:
    inferred_task = "Classification"
 
st.subheader("🧠 Prediction Type")
 
task_type = st.radio(
    "Choose the type of prediction",
    options=["Automatic", "Classification", "Regression"],
    index=0,
    horizontal=True,
    help=(
        "Automatic makes a best-effort guess from the selected "
        "target. You can override it if your task requires it."
    ),
    key="dataset_task_type",
)
 
if task_type == "Automatic":
    task = inferred_task
else:
    task = task_type
 
st.info(f"Selected task: **{task}**")
 
# ============================================================
# TARGET INFORMATION
# ============================================================
 
tc1, tc2, tc3 = st.columns(3)
 
tc1.metric("Target Column", str(target_column))
tc2.metric("Unique Values", f"{unique_count:,}")
tc3.metric(
    "Missing Target Values",
    f"{int(raw_target.isna().sum()):,}",
)
 
if task == "Classification":
    st.write("#### Target Class Distribution")
    distribution = (
        raw_target.value_counts(dropna=False)
        .rename_axis("Class")
        .reset_index(name="Count")
    )
    st.dataframe(distribution, use_container_width=True)
 
# ============================================================
# PREPARE WORKING DATA
# ============================================================
 
working_df = df.copy()
 
# The target must be present.
working_df = working_df.dropna(subset=[target_column]).copy()
 
if task == "Regression":
    working_df[target_column] = pd.to_numeric(
        working_df[target_column], errors="coerce"
    )
    working_df = working_df.dropna(subset=[target_column]).copy()
 
    working_df = working_df[
        np.isfinite(working_df[target_column].astype(float))
    ].copy()
 
else:
    # Preserve class labels as strings for classification.
    working_df[target_column] = (
        working_df[target_column].astype(str)
    )
 
if len(working_df) < 4:
    st.error(
        "Too few usable rows remain. Upload more data or "
        "choose a different target column."
    )
    st.stop()
 
# ============================================================
# IDENTIFIER AND LEAKAGE PROTECTION
# ============================================================
 
identifier_columns = []
 
for column in working_df.columns:
    if column == target_column:
        continue
 
    name = str(column).strip().lower()
 
    if (
        name == "id"
        or name.endswith("_id")
        or name.endswith(" id")
    ):
        identifier_columns.append(column)
 
leakage_columns = []
 
# These columns may reveal information unavailable at prediction time
# for the specific student dropout target.
if target_column == "Target_Dropout_Next_Sem":
    for column in ["End_of_Semester_Status", "Censored"]:
        if column in working_df.columns:
            leakage_columns.append(column)
 
if identifier_columns:
    st.warning(
        "Identifier columns excluded from model inputs: "
        + ", ".join(map(str, identifier_columns))
    )
 
if leakage_columns:
    st.warning(
        "Potential outcome/leakage columns excluded: "
        + ", ".join(map(str, leakage_columns))
    )
 
columns_to_drop = list(
    dict.fromkeys(
        [target_column] + identifier_columns + leakage_columns
    )
)
 
X_raw = working_df.drop(columns=columns_to_drop).copy()
y = working_df[target_column].copy()
 
# Remove completely empty input features.
empty_columns = [
    col for col in X_raw.columns if X_raw[col].isna().all()
]
 
if empty_columns:
    st.warning(
        "Completely empty features were removed: "
        + ", ".join(map(str, empty_columns))
    )
    X_raw = X_raw.drop(columns=empty_columns)
 
if X_raw.shape[1] == 0:
    st.error(
        "No usable input features remain after excluding the target, "
        "identifiers, and leakage columns."
    )
    st.stop()
 
# ============================================================
# PREPROCESS FEATURES
# ============================================================
 
X = X_raw.copy()
 
numeric_columns = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()
 
categorical_columns = [
    col for col in X.columns if col not in numeric_columns
]
 
# ------------------------------------------------------------
# Exclude high-cardinality text columns (URLs, titles, dates,
# names, free text). One-hot encoding them would create thousands
# of columns and can exhaust the memory of the server.
# ------------------------------------------------------------
 
MAX_CATEGORIES = 30
 
high_cardinality_columns = [
    col for col in categorical_columns
    if X[col].nunique(dropna=True) > MAX_CATEGORIES
]
 
if high_cardinality_columns:
    st.warning(
        f"Text columns with more than {MAX_CATEGORIES} distinct values "
        "were excluded from model inputs: "
        + ", ".join(map(str, high_cardinality_columns))
    )
 
    X = X.drop(columns=high_cardinality_columns)
 
    categorical_columns = [
        col for col in categorical_columns
        if col not in high_cardinality_columns
    ]
 
if X.shape[1] == 0:
    st.error(
        "No usable input features remain after excluding text columns "
        "with too many distinct values. Choose another dataset or target."
    )
    st.stop()
 
# Fill numerical missing values with the median.
for col in numeric_columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].replace([np.inf, -np.inf], np.nan)
 
    median = X[col].median()
 
    if pd.isna(median):
        median = 0
 
    X[col] = X[col].fillna(median)
 
# Fill categorical missing values and convert to text.
for col in categorical_columns:
    X[col] = X[col].astype("string").fillna("Unknown").astype(str)
 
# One-hot encode categorical features.
if categorical_columns:
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=int,
    )
 
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
 
# ============================================================
# VALIDATE TARGET
# ============================================================
 
if task == "Classification":
    y = y.astype(str)
 
    if y.nunique() < 2:
        st.error(
            "Classification requires at least two distinct target classes. "
            "Choose another target or select Regression if appropriate."
        )
        st.stop()
 
    class_counts = y.value_counts()
 
    if class_counts.min() < 2:
        st.warning(
            "At least one class has only one sample. "
            "The app cannot safely create a stratified split for this target. "
            "Choose another target or provide more examples per class."
        )
        st.stop()
 
else:
    y = pd.to_numeric(y, errors="coerce")
    valid_mask = y.notna() & np.isfinite(y.astype(float))
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
 
    if y.nunique() < 2:
        st.error(
            "Regression requires at least two distinct numerical target values."
        )
        st.stop()
 
# ============================================================
# FEATURE INFORMATION
# ============================================================
 
st.subheader("🧩 Feature Information")
 
fc1, fc2, fc3 = st.columns(3)
 
fc1.metric("Input Features", f"{X.shape[1]:,}")
fc2.metric("Numerical Source Features", f"{len(numeric_columns):,}")
fc3.metric("Categorical Source Features", f"{len(categorical_columns):,}")
 
with st.expander("View input feature names"):
    st.write(X.columns.tolist())
 
st.subheader("⚙️ Prepared Dataset")
 
pc1, pc2, pc3 = st.columns(3)
pc1.metric("Usable Samples", f"{len(X):,}")
pc2.metric("Features After Encoding", f"{X.shape[1]:,}")
pc3.metric(
    "Target Classes" if task == "Classification" else "Unique Target Values",
    f"{y.nunique():,}",
)
 
# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
 
st.subheader("🔀 Train/Test Split")
 
test_fraction = 0.20
 
stratify_y = None
 
if task == "Classification":
    # Stratification requires enough samples in each class for both splits.
    n_classes = y.nunique()
    n_test = int(np.ceil(len(y) * test_fraction))
    n_train = len(y) - n_test
 
    if (
        y.value_counts().min() >= 2
        and n_test >= n_classes
        and n_train >= n_classes
    ):
        stratify_y = y
 
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=42,
        stratify=stratify_y,
    )
except ValueError as e:
    st.error(f"Unable to split the dataset: {e}")
    st.stop()
 
sc1, sc2 = st.columns(2)
sc1.metric("Training Samples", f"{len(X_train):,}")
sc2.metric("Testing Samples", f"{len(X_test):,}")
 
# ============================================================
# TRAIN MODEL BUTTON
# ============================================================
 
st.subheader("🚀 Model Training")
 
st.write(
    f"The model will learn to predict **{target_column}** "
    f"using **{task.lower()}**."
)
 
train_clicked = st.button(
    "🚀 Train Model",
    type="primary",
    use_container_width=True,
    key="dataset_train_button",
)
 
# ============================================================
# TRAIN MODEL
# ============================================================
 
if train_clicked:
    try:
        with st.spinner(f"Training Random Forest {task.lower()} model..."):
            if task == "Classification":
                model = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                )
 
            model.fit(X_train, y_train)
 
            y_pred = model.predict(X_test)
 
            results = {
                "model": model,
                "task": task,
                "target": target_column,
                "feature_names": X.columns.tolist(),
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred": y_pred,
                "dataset_signature": dataset_signature,
            }
 
            st.session_state.dataset_model = results
            st.session_state.dataset_model_trained = True
 
    except Exception as e:
        st.session_state.dataset_model_trained = False
        st.error(f"Model training failed: {e}")
 
# ============================================================
# DISPLAY TRAINED MODEL RESULTS
# ============================================================
 
results = st.session_state.get("dataset_model")
 
if (
    not st.session_state.dataset_model_trained
    or results is None
    or results.get("target") != target_column
    or results.get("task") != task
    or results.get("dataset_signature") != dataset_signature
):
    st.info(
        "Choose your target column and click **Train Model** "
        "to see the results."
    )
    st.stop()
 
model = results["model"]
X_test = results["X_test"]
y_test = results["y_test"]
y_pred = results["y_pred"]
 
st.success(
    f"Model trained successfully for target: {target_column}"
)
 
# ============================================================
# PERFORMANCE METRICS
# ============================================================
 
st.subheader("📊 Model Performance")
 
if task == "Classification":
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )
 
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    mc2.metric("Macro Precision", f"{precision * 100:.2f}%")
    mc3.metric("Macro Recall", f"{recall * 100:.2f}%")
    mc4.metric("Macro F1 Score", f"{f1 * 100:.2f}%")
 
    st.caption(
        "These metrics measure classification performance on the held-out test set."
    )
 
    st.subheader("🔲 Confusion Matrix")
 
    classes = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
 
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {c}" for c in classes],
        columns=[f"Predicted: {c}" for c in classes],
    )
 
    st.dataframe(cm_df, use_container_width=True)
 
else:
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred) if len(y_test) >= 2 else float("nan")
 
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Mean Absolute Error", f"{mae:.4f}")
    mc2.metric("Root Mean Squared Error", f"{rmse:.4f}")
    mc3.metric("R² Score", f"{r2:.4f}" if np.isfinite(r2) else "N/A")
 
    st.caption(
        "Lower MAE and RMSE generally indicate smaller prediction errors. "
        "R² can be negative when predictions perform poorly."
    )
 
    comparison_df = pd.DataFrame({
        "Actual": np.asarray(y_test),
        "Predicted": np.asarray(y_pred),
    })
 
    st.subheader("📈 Actual vs Predicted")
    st.dataframe(comparison_df.head(100), use_container_width=True)
 
# ============================================================
# FEATURE IMPORTANCE
# ============================================================
 
st.subheader("🔍 Feature Importance")
 
importance_df = pd.DataFrame({
    "Feature": results["feature_names"],
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)
 
st.dataframe(importance_df.head(20), use_container_width=True)
 
st.bar_chart(
    importance_df.head(15).set_index("Feature")["Importance"]
)
 
csv_data = importance_df.to_csv(index=False).encode("utf-8")
 
st.download_button(
    "⬇️ Download Feature Importance (CSV)",
    data=csv_data,
    file_name="feature_importance.csv",
    mime="text/csv",
)
 
# ============================================================
# SAMPLE PREDICTION
# ============================================================
 
st.subheader("🔮 Test Sample Prediction")
 
if len(X_test) > 0:
    sample_index = st.number_input(
        "Choose a test sample index",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1,
        key="dataset_sample_index",
    )
 
    selected_sample = X_test.iloc[[int(sample_index)]]
    actual_value = y_test.iloc[int(sample_index)]
    predicted_value = model.predict(selected_sample)[0]
 
    p1, p2 = st.columns(2)
    p1.metric("Actual Target", str(actual_value))
    p2.metric("Predicted Target", str(predicted_value))
 
    if task == "Classification":
        probabilities = model.predict_proba(selected_sample)[0]
 
        probability_df = pd.DataFrame({
            "Class": [str(c) for c in model.classes_],
            "Probability": [
                f"{p * 100:.2f}%" for p in probabilities
            ],
        })
 
        st.write("#### Class Probabilities")
        st.dataframe(probability_df, use_container_width=True)
 
    with st.expander("View input features for this prediction"):
        st.dataframe(selected_sample.T, use_container_width=True)
 
# ============================================================
# XAI EXPLANATION
# ============================================================
 
st.subheader("💡 Explainable AI")
 
st.info(
    """
    **Global explanation:** Feature importance ranks the features
    the trained Random Forest uses across the model.
 
    **Prediction explanation:** The sample prediction shows the
    model's output for one test record.
 
    Feature importance describes model behavior. It does not prove
    that a feature causes the target outcome, and it does not guarantee
    that the model is correct.
    """
)
 
st.caption(
    f"Model: Random Forest {task} | Trees: 200 | "
    "Train/Test Split: 80/20 | Random State: 42"
)
 
