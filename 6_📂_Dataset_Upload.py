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
