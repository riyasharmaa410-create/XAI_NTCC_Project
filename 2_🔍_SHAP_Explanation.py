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
