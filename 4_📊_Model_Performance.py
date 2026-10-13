
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)

st.title("📈 Model Performance")
st.write(
    "Evaluate the trained machine learning model using "
    "the test data from your uploaded dataset."
)

st.divider()

# --------------------------------------------------
# 1. GET THE TRAINED MODEL
# --------------------------------------------------

dataset_model = st.session_state.get("dataset_model")
model_trained = st.session_state.get(
    "dataset_model_trained", False
)

if not model_trained or not isinstance(dataset_model, dict):
    st.warning("No uploaded dataset model is currently active.")
    st.info(
        "Go to 📂 Dataset Upload, upload your dataset, "
        "select the target column, and train the model first."
    )
    st.stop()

model = dataset_model.get("model")
task = str(dataset_model.get("task", "")).lower()

X_test = dataset_model.get("X_test")
y_test = dataset_model.get("y_test")
y_pred = dataset_model.get("y_pred")

if model is None or X_test is None or y_test is None:
    st.error(
        "The model or test data is missing. Please train "
        "the model again on the Dataset Upload page."
    )
    st.stop()

# Convert test data into a DataFrame if needed.
feature_names = dataset_model.get("feature_names", [])

if not isinstance(X_test, pd.DataFrame):
    if feature_names and len(feature_names) == np.asarray(X_test).shape[1]:
        X_test = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_test = pd.DataFrame(X_test)

# Recalculate predictions to avoid stale predictions.
try:
    y_pred = model.predict(X_test)
except Exception as error:
    st.error("The model could not predict on the saved test data.")
    st.code(str(error))
    st.stop()

y_test = np.asarray(y_test)
y_pred = np.asarray(y_pred)

if len(y_test) != len(y_pred):
    st.error("The test labels and predictions have different lengths.")
    st.stop()

# --------------------------------------------------
# 2. CLASSIFICATION PERFORMANCE
# --------------------------------------------------

if task == "classification":
    st.subheader("🎯 Classification Metrics")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Precision", f"{precision * 100:.2f}%")
    col3.metric("Recall", f"{recall * 100:.2f}%")
    col4.metric("F1 Score", f"{f1 * 100:.2f}%")

    st.caption(
        "Weighted averages account for the number of test samples "
        "in each class. Results depend on your dataset and split."
    )

    st.divider()
    st.subheader("📊 Confusion Matrix")

    labels = np.unique(np.concatenate((y_test, y_pred)))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in labels],
        columns=[f"Predicted: {label}" for label in labels]
    )

    st.dataframe(matrix_df, use_container_width=True)

    st.bar_chart(matrix_df)

    st.divider()
    st.subheader("📝 Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

# --------------------------------------------------
# 3. REGRESSION PERFORMANCE
# --------------------------------------------------

elif task == "regression":
    st.subheader("📉 Regression Metrics")

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("MSE", f"{mse:.4f}")
    col3.metric("RMSE", f"{rmse:.4f}")
    col4.metric("R² Score", f"{r2:.4f}")

    st.divider()
    st.subheader("📊 Actual vs Predicted")

    comparison_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    st.dataframe(
        comparison_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.line_chart(comparison_df)

    st.divider()
    st.subheader("📏 Prediction Errors")

    comparison_df["Error"] = (
        comparison_df["Actual"] - comparison_df["Predicted"]
    )

    st.bar_chart(comparison_df["Error"])

else:
    st.error(
        f"Unsupported task type: {task!r}. "
        "Please select Classification or Regression and train again."
    )
    st.stop()

# --------------------------------------------------
# 4. GENERAL INFORMATION
# --------------------------------------------------

st.divider()
st.subheader("ℹ️ Evaluation Information")

st.write(f"**Task:** {task.title()}")
st.write(f"**Number of test samples:** {len(y_test)}")
st.write(f"**Number of features:** {X_test.shape[1]}")

st.success(
    "Performance metrics were calculated using the test data "
    "saved by the Dataset Upload page."
)
