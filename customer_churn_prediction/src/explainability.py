import joblib
import shap
from shap import maskers
import matplotlib.pyplot as plt

# load model and data

model = joblib.load('artifacts/model.joblib')
X_test = joblib.load('artifacts/X_test.joblib')

# Extract components from pipeline

preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

# Transform data
X_transformed = preprocessor.transform(X_test)

# Get feature names
features_names = preprocessor.get_feature_names_out()

# ...existing code...
# Modern SHAP Explainer (no warning)
explainer = shap.LinearExplainer(
    classifier,
    maskers.Independent(X_transformed), # Use the new masker syntax
)                 # This returns an Explanation object

explanation = explainer(X_transformed) # Call explainer like a function now

# Access the attributes using .values and .base_values
shap_values = explanation.values                       
expected_value = explanation.base_values

# Summary plot - can accept the full explanation object now
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=features_names,
    show=False
)

plt.savefig('artifacts/shap_summary_plot.png', bbox_inches='tight')
plt.close()

# Local explanation for one customer
idx = 0

# You can simplify this logic now that you have the explanation object
sv = explanation[idx] # Slice the explanation object for the single index

shap.force_plot(
    sv.base_values,
    sv.values,
    feature_names=features_names,
    matplotlib=True
)
plt.savefig("artifacts/shap_local_example.png", bbox_inches="tight")
plt.close()
# ...existing code..