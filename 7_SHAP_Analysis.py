import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📈 SHAP Analysis")

st.write(
    """
    SHAP (SHapley Additive exPlanations) shares a model's prediction
    among the input features using Shapley values from cooperative
    game theory. For a Random Forest, the exact **TreeSHAP** algorithm
    is used. This page complements the **Feature Explanation** page,
    which uses a different, perturbation-based method.
    """
)


# ============================================================
# IMPORT SHAP (the page still opens if SHAP cannot load)
# ============================================================

try:
    import shap
except Exception as error:  # ImportError, or a blocked DLL (OSError)
    st.error(
        "SHAP could not be loaded in this environment, so SHAP "
        "values cannot be calculated here."
    )
    st.caption(f"Technical message: {error}")
    st.info(
        "The **Feature Explanation** page does not need SHAP and "
        "remains available."
    )
    st.stop()


# ============================================================
# LOAD THE ACTIVE MODEL
# (uploaded model if trained, otherwise the saved WDBC model)
# ============================================================

def find_model_folder():
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here)):
        candidate = os.path.join(base, "model")
        if os.path.isdir(candidate):
            return candidate
    return None


uploaded_available = (
    st.session_state.get("model_trained", False)
    and "uploaded_model" in st.session_state
)

if uploaded_available:

    model = st.session_state["uploaded_model"]
    X_test = st.session_state["uploaded_X_test"]
    feature_names = list(st.session_state["uploaded_feature_names"])
    class_names = [str(c) for c in st.session_state["uploaded_classes"]]
    target_column = st.session_state.get("uploaded_target_column", "Target")
    model_source = "Uploaded dataset model"

else:

    model_folder = find_model_folder()

    if model_folder is None:
        st.error("The saved model folder could not be found.")
        st.stop()

    model = joblib.load(
        os.path.join(model_folder, "xai_random_forest_model.pkl")
    )
    X_test = pd.read_csv(os.path.join(model_folder, "X_test.csv"))
    feature_names = list(
        joblib.load(os.path.join(model_folder, "feature_names.pkl"))
    )
    class_names = ["Benign", "Malignant"]
    target_column = "diagnosis"
    model_source = "Original WDBC model"

X_test = pd.DataFrame(X_test).reset_index(drop=True)
model_classes = list(model.classes_)

st.caption(
    f"Active model: **{model_source}** · Target: `{target_column}` · "
    f"Features: {len(feature_names)} · Test samples: {len(X_test)}"
)

st.markdown("---")


# ============================================================
# SHAP HELPERS
# ============================================================

def class_slice(values, class_index):
    """
    Return a (samples x features) array for one class.

    Different SHAP versions return a list of arrays (one per class)
    or a 3-D array (samples x features x classes).
    """
    if isinstance(values, list):
        return np.asarray(values[class_index])

    values = np.asarray(values)

    if values.ndim == 3:
        return values[:, :, class_index]

    return values


def class_base_value(expected_value, class_index):
    expected_value = np.asarray(expected_value).ravel()

    if expected_value.size > class_index:
        return float(expected_value[class_index])

    return float(expected_value[0])


@st.cache_resource(show_spinner=False)
def get_explainer(_model, cache_key):
    return shap.TreeExplainer(_model)


def compute_shap(explainer, data):
    return explainer.shap_values(data, check_additivity=False)


cache_key = (
    model_source,
    len(feature_names),
    len(X_test),
    str(target_column),
)

try:
    explainer = get_explainer(model, cache_key)
except Exception as error:
    st.error("A SHAP explainer could not be created for this model.")
    st.caption(f"Technical message: {error}")
    st.stop()


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

st.header("🌍 Global SHAP Importance")

st.write(
    """
    The mean absolute SHAP value shows how strongly each feature moves
    the model's output **on average**. Larger values mean the feature
    changes predictions more across the sample used below.
    """
)

class_choice = st.selectbox(
    "Class to explain",
    options=list(range(len(model_classes))),
    index=len(model_classes) - 1,
    format_func=lambda i: (
        f"{class_names[i]} (class {model_classes[i]})"
        if i < len(class_names)
        else str(model_classes[i])
    ),
)

max_rows = min(300, len(X_test))

sample_size = st.slider(
    "Number of test samples used for the global summary",
    min_value=min(20, max_rows),
    max_value=max_rows,
    value=max_rows,
)

global_data = X_test.iloc[:sample_size]

with st.spinner("Calculating SHAP values..."):
    global_values = class_slice(
        compute_shap(explainer, global_data),
        class_choice,
    )

mean_abs = np.abs(global_values).mean(axis=0)

importance = (
    pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
    .sort_values("Mean |SHAP|", ascending=False)
    .reset_index(drop=True)
)

top_n = min(15, len(importance))
top = importance.head(top_n).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 0.38 * top_n + 1.2))
ax.barh(top["Feature"], top["Mean |SHAP|"], color="#1F5FA8")
ax.set_xlabel("Mean absolute SHAP value")
ax.set_title(
    f"Top {top_n} features for class: {class_names[class_choice]}"
)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.dataframe(importance.head(top_n), width="stretch", hide_index=True)

st.caption(
    f"Calculated on {sample_size} test samples. Values are in "
    "probability units of the selected class."
)

st.markdown("---")


# ============================================================
# LOCAL SHAP EXPLANATION
# ============================================================

st.header("🔍 Local SHAP Explanation")

sample_index = st.number_input(
    "Test sample index",
    min_value=0,
    max_value=len(X_test) - 1,
    value=0,
    step=1,
)

sample = X_test.iloc[[int(sample_index)]]

probabilities = model.predict_proba(sample)[0]
predicted_position = int(np.argmax(probabilities))

local_values = class_slice(compute_shap(explainer, sample), class_choice)[0]
base_value = class_base_value(explainer.expected_value, class_choice)

model_output = float(probabilities[class_choice])
shap_total = float(local_values.sum())

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Predicted class",
        class_names[predicted_position]
        if predicted_position < len(class_names)
        else str(model_classes[predicted_position]),
    )

with c2:
    st.metric(
        f"P({class_names[class_choice]})",
        f"{model_output * 100:.2f}%",
    )

with c3:
    st.metric(
        "Base value (average)",
        f"{base_value * 100:.2f}%",
    )

local_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Value": sample.iloc[0].values,
        "SHAP value": local_values,
    }
)
local_df["Abs"] = local_df["SHAP value"].abs()
local_df = local_df.sort_values("Abs", ascending=False)

top_local = local_df.head(10).iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 4.6))
colors = [
    "#C8731A" if value >= 0 else "#1F5FA8"
    for value in top_local["SHAP value"]
]
ax.barh(top_local["Feature"], top_local["SHAP value"], color=colors)
ax.axvline(0, color="#444444", linewidth=0.8)
ax.set_xlabel(
    f"SHAP value (effect on P({class_names[class_choice]}))"
)
ax.set_title(f"Local SHAP values, test sample {int(sample_index)}")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Orange bars push the probability of the selected class up; "
    "blue bars push it down."
)

st.dataframe(
    local_df.drop(columns="Abs").head(15),
    width="stretch",
    hide_index=True,
)

st.write(
    f"""
    **Additivity check:** base value ({base_value:.4f}) + sum of SHAP
    values ({shap_total:+.4f}) = **{base_value + shap_total:.4f}**,
    compared with the model's probability **{model_output:.4f}**.
    """
)

st.markdown("---")


# ============================================================
# HOW TO READ THIS PAGE
# ============================================================

st.subheader("💡 How to read SHAP values")

st.write(
    """
    Each SHAP value is the amount a feature moves the prediction away
    from the average prediction (the base value) for this sample. The
    base value plus all SHAP values equals the model's output.
    """
)

st.info(
    "SHAP and the perturbation-based Feature Explanation page answer "
    "different questions and can rank features differently. SHAP "
    "describes how this model uses the features; it does not prove "
    "that a feature causes the real-world outcome."
)

if uploaded_available:
    st.warning(
        "For an uploaded dataset, categorical columns have been "
        "one-hot encoded, so SHAP values are reported per encoded "
        "feature."
    )
