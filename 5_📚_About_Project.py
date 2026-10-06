import streamlit as st


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📚 About the Project")

st.markdown(
    """
# Explainable AI (XAI): Making Artificial Intelligence Transparent

This project demonstrates how Explainable Artificial Intelligence (XAI)
can make machine learning predictions easier for humans to understand.

The application combines machine learning prediction with explanation
and interactive analysis so that users can explore not only what a model
predicts, but also how changes in input features can affect its behaviour.
"""
)


# ============================================================
# PROJECT OBJECTIVE
# ============================================================

st.header("🎯 Project Objective")

st.markdown(
    """
The main objective of this project is to develop an interactive
Explainable AI application that can:

- Train a machine learning classification model
- Generate predictions and prediction probabilities
- Explain the contribution of input features
- Perform What-If analysis by changing individual features
- Display model performance using standard evaluation metrics
- Allow users to upload their own suitable tabular datasets
"""
)


# ============================================================
# CURRENT DATASET
# ============================================================

st.header("📊 Current Dataset")

st.markdown(
    """
The current working demonstration uses an uploaded academic tabular
dataset containing student-related information and a dropout prediction
target.

The active target used in the current demonstration is:

**Target_Dropout_Next_Sem**

Outcome-related columns such as **End_of_Semester_Status** and
**Censored** are excluded when predicting the dropout target in order
to reduce the possibility of data leakage.

The uploaded dataset contains numerical and categorical variables.
Categorical variables are converted into numerical form before being
provided to the Random Forest model.
"""
)


# ============================================================
# DATASET UPLOAD
# ============================================================

st.header("📂 Dataset Upload Functionality")

st.markdown(
    """
A Dataset Upload feature has been added to make the application more
flexible.

Users can upload a tabular dataset, inspect its structure, select a
suitable target column, preprocess the available features and train a
Random Forest classification model.

The application also identifies certain outcome-related columns when
predicting **Target_Dropout_Next_Sem** and excludes them from the model
inputs to reduce potential data leakage.
"""
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

st.header("🤖 Machine Learning Model")

st.markdown(
    """
A Random Forest Classifier is used as the primary machine learning
model in the application.

- **Number of trees:** 200
- **Training/Test split:** 80% / 20%
- **Random state:** 42

The model is trained on the training portion of the dataset and
evaluated on the held-out test dataset.
"""
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📈 Current Model Performance")

st.markdown(
    """
For the current uploaded student dataset, the trained Random Forest
model produced the following results on the held-out test dataset:

- **Accuracy:** 91.62%
- **Macro Precision:** 78.40%
- **Macro Recall:** 54.11%
- **Macro F1 Score:** 55.45%

These results describe the performance of the current trained model
on its held-out test dataset. They should not be assumed to represent
performance on other datasets.
"""
)


# ============================================================
# EXPLAINABLE AI TECHNIQUE
# ============================================================

st.header("🔍 Explainable AI Technique")

st.markdown(
    """
The application provides a model-based local feature contribution
method to explore why a particular prediction was produced.

For a selected test sample, the application compares the model's
original prediction probability with predictions obtained after
replacing individual feature values with reference values.

This allows the user to identify features that have a stronger effect
on the model's output for the selected sample.

These contribution values describe the behaviour of the trained
machine learning model. They should not be interpreted as proof of
real-world causation.
"""
)


# ============================================================
# SHAP TECHNICAL NOTE
# ============================================================

st.header("📝 SHAP Technical Note")

st.markdown(
    """
SHAP was part of the original XAI implementation of this project.
However, in the current Windows environment, the SHAP dependency
chain requires Numba components that are blocked by the system's
Windows Application Control policy.

Therefore, the current web application uses a Numba-free,
model-based feature contribution approach instead of displaying
the resulting values as SHAP values.

The current feature contribution values should not be called
SHAP values. They are model-based local contribution estimates.
"""
)


# ============================================================
# APPLICATION FEATURES
# ============================================================

st.header("⚙️ Application Features")


st.subheader("1. 📂 Dataset Upload")

st.markdown(
    """
Allows users to upload a suitable tabular classification dataset,
select the target column and train a Random Forest model.
"""
)


st.subheader("2. 🔮 Prediction")

st.markdown(
    """
Generates predictions using the trained Random Forest model and
displays the predicted class and prediction probabilities.
"""
)


st.subheader("3. 🔍 Feature Explanation")

st.markdown(
    """
Provides a local model-based explanation showing which features
contribute to the selected prediction.
"""
)


st.subheader("4. 🧪 What-If Analysis")

st.markdown(
    """
Allows a feature value to be changed while observing how the
model's prediction probability responds.
"""
)


st.subheader("5. 📊 Model Performance")

st.markdown(
    """
Displays Accuracy, Macro Precision, Macro Recall, Macro F1 Score,
Confusion Matrix and test-set class distribution.
"""
)


st.subheader("6. 📚 About Project")

st.markdown(
    """
Provides information about the project objective, methodology,
current dataset, technologies, limitations and future scope.
"""
)


# ============================================================
# TECHNOLOGIES USED
# ============================================================

st.header("🛠️ Technologies Used")

st.markdown(
    """
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Joblib
- Matplotlib

SHAP was used during the earlier implementation stage, but the
current deployed explanation page uses the model-based feature
contribution approach because of the local Numba DLL restriction.
"""
)


# ============================================================
# LIMITATIONS
# ============================================================

st.header("⚠️ Limitations")

st.markdown(
    """
- The application is primarily designed for educational and research
  demonstration purposes.
- Model performance depends on the quality and characteristics of the
  dataset used for training.
- Feature contribution explanations describe model behaviour and do
  not establish causation.
- The current Dataset Upload functionality focuses on tabular
  classification datasets.
- The current application uses Random Forest as the primary
  classification algorithm.
- The current local explanation method is not equivalent to SHAP and
  should not be presented as SHAP output.
"""
)


# ============================================================
# FUTURE SCOPE
# ============================================================

st.header("🚀 Future Scope")

st.markdown(
    """
Future improvements can include:

- Support for additional machine learning algorithms
- Model comparison functionality
- Additional explainability techniques such as SHAP and LIME
- More advanced global and local explanation visualizations
- Downloadable prediction and explanation reports
- Improved interactive visualizations
- Support for regression problems
- Automated dataset quality and leakage detection
- More advanced preprocessing options
- Support for larger varieties of tabular datasets
"""
)


# ============================================================
# PROJECT SUMMARY
# ============================================================

st.header("📌 Project Summary")

st.markdown(
    """
This project demonstrates an end-to-end Explainable AI workflow:

**Dataset Upload → Data Preprocessing → Model Training → Prediction
→ Feature Explanation → What-If Analysis → Model Performance Evaluation**

The main purpose of the application is to make machine learning
models more understandable by showing not only what the model
predicts, but also how its predictions respond to changes in input
features.
"""
)


# ============================================================
# DISCLAIMER
# ============================================================

st.info(
    """
This application is intended for educational and research
demonstration purposes. Model predictions and explanations should be
interpreted within the context of the dataset and model used.
Feature contributions describe model behaviour and do not by
themselves establish real-world causation.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "**NTCC Project | Explainable AI: Making AI Transparent**"
)