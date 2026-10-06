import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Explainable AI",
    page_icon="🤖",
    layout="wide"
)


# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------
def dashboard():

    # Check whether a dataset model has been trained
    model_trained = st.session_state.get("model_trained", False)

    if model_trained:

        model = st.session_state.get("uploaded_model")
        X_test = st.session_state.get("uploaded_X_test")
        feature_names = st.session_state.get(
            "uploaded_feature_names",
            []
        )
        target_column = st.session_state.get(
            "uploaded_target_column",
            "Target"
        )

        test_samples = len(X_test)
        feature_count = len(feature_names)

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------
        st.title("🤖 Explainable AI")
        st.subheader("Making Artificial Intelligence Transparent")

        st.write(
            """
            This application demonstrates how Explainable AI (XAI)
            can make machine learning predictions easier to understand.
            Instead of only showing the final prediction, the system
            also provides information about the features that influenced
            the prediction.
            """
        )

        st.divider()

        # -------------------------------------------------
        # MODEL SUMMARY
        # -------------------------------------------------
        st.subheader("📌 Current Model")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Model",
                "Random Forest"
            )

        with col2:
            st.metric(
                "Test Samples",
                f"{test_samples:,}"
            )

        with col3:
            st.metric(
                "Input Features",
                feature_count
            )

        with col4:
            st.metric(
                "Number of Trees",
                200
            )

        st.info(
            f"🎯 Active prediction target: **{target_column}**"
        )

        st.divider()

        # -------------------------------------------------
        # WHAT IS XAI?
        # -------------------------------------------------
        st.subheader("🔍 What is Explainable AI?")

        st.write(
            """
            Explainable Artificial Intelligence (XAI) refers to methods
            that help humans understand how a machine learning model
            reaches its predictions.
            """
        )

        st.write(
            """
            A traditional machine learning model may provide an output
            such as:
            """
        )

        st.code(
            "Prediction: Student is likely to continue studies"
        )

        st.write(
            """
            XAI goes one step further by helping answer:
            """
        )

        st.success(
            "“Why did the model make this prediction?”"
        )

        st.write(
            """
            In this project, feature contribution analysis is used to
            show how individual input features affect the model's
            prediction.
            """
        )

        st.divider()

        # -------------------------------------------------
        # PROJECT WORKFLOW
        # -------------------------------------------------
        st.subheader("🔄 Project Workflow")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### 📂 1. Dataset")
            st.write(
                """
                Upload a dataset and select the target column
                that the model should predict.
                """
            )

        with col2:
            st.markdown("### 🔮 2. Prediction")
            st.write(
                """
                A Random Forest model is trained and used
                to generate predictions.
                """
            )

        with col3:
            st.markdown("### 🔍 3. Explanation")
            st.write(
                """
                Feature contribution analysis explains which
                input features influenced the prediction.
                """
            )

        with col4:
            st.markdown("### 🧪 4. What-If")
            st.write(
                """
                Change an input feature and observe how the
                model's prediction changes.
                """
            )

        st.divider()

        # -------------------------------------------------
        # CURRENT PROJECT STATUS
        # -------------------------------------------------
        st.subheader("📊 Current Project Status")

        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            st.success("✅ Prediction Available")

        with status_col2:
            st.success("✅ Feature Explanation Available")

        with status_col3:
            st.success("✅ What-If Analysis Available")

        st.divider()

        # -------------------------------------------------
        # MODEL INFORMATION
        # -------------------------------------------------
        st.subheader("⚙️ Model Information")

        model_information = {
            "Algorithm": "Random Forest Classifier",
            "Number of Trees": "200",
            "Train/Test Split": "80% / 20%",
            "Random State": "42",
            "Input Features": str(feature_count),
            "Test Samples": str(test_samples),
            "Target Column": target_column
        }

        st.table(model_information)

        st.divider()

        # -------------------------------------------------
        # XAI FEATURES
        # -------------------------------------------------
        st.subheader("✨ XAI Features")

        st.markdown(
            """
            **🔮 Prediction**
            
            Generate predictions using the trained machine learning model.

            **🔍 Feature Explanation**
            
            Understand which features contributed to an individual
            model prediction.

            **🧪 What-If Analysis**
            
            Change a feature value and observe how the model's
            output changes.

            **📊 Model Performance**
            
            View accuracy, precision, recall, F1-score and the
            confusion matrix.

            **📂 Dataset Upload**
            
            Upload your own compatible dataset and train a new model.
            """
        )

        st.divider()

        # -------------------------------------------------
        # IMPORTANT NOTE
        # -------------------------------------------------
        st.info(
            """
            **Note:** The Feature Explanation page currently uses a
            model-based local feature contribution method. It does not
            display SHAP values because the current Windows environment
            blocks a required Numba component.
            """
        )

    else:

        # -------------------------------------------------
        # INITIAL DASHBOARD
        # -------------------------------------------------
        st.title("🤖 Explainable AI")
        st.subheader("Making Artificial Intelligence Transparent")

        st.write(
            """
            Welcome to the Explainable AI web application.

            This project demonstrates how machine learning predictions
            can be made easier to understand using Explainable AI
            techniques.
            """
        )

        st.divider()

        st.subheader("🚀 Getting Started")

        st.info(
            """
            No model is currently active.

            Go to **📂 Dataset Upload** from the sidebar to upload
            a dataset, select the target column, and train the
            Random Forest model.
            """
        )

        st.divider()

        st.subheader("🔍 What this project demonstrates")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📊 Machine Learning")
            st.write(
                """
                Train a Random Forest classification model
                on an uploaded dataset.
                """
            )

        with col2:
            st.markdown("### 🔍 Explainability")
            st.write(
                """
                Understand which features influence individual
                predictions.
                """
            )

        with col3:
            st.markdown("### 🧪 What-If Analysis")
            st.write(
                """
                Modify feature values and observe how the model
                responds.
                """
            )

        st.divider()

        st.subheader("📂 Start Here")

        st.write(
            """
            Use the **Dataset Upload** page from the sidebar
            to begin the project workflow.
            """
        )


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

pages = [
    st.Page(
        dashboard,
        title="Dashboard",
        icon="🏠"
    ),

    st.Page(
        "pages/1_🔮_Prediction.py",
        title="Prediction",
        icon="🔮"
    ),

    st.Page(
        "pages/2_🔍_SHAP_Explanation.py",
        title="Feature Explanation",
        icon="🔍"
    ),

    st.Page(
        "pages/3_🧪_What_If_Analysis.py",
        title="What-If Analysis",
        icon="🧪"
    ),

    st.Page(
        "pages/4_📊_Model_Performance.py",
        title="Model Performance",
        icon="📊"
    ),

    st.Page(
        "pages/5_📚_About_Project.py",
        title="About Project",
        icon="📚"
    ),

    st.Page(
        "pages/6_📂_Dataset_Upload.py",
        title="Dataset Upload",
        icon="📂"
    )
]


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------

pg = st.navigation(pages)

pg.run()