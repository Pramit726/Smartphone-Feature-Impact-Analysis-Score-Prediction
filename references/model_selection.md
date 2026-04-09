# Model Selection Guide for Regression

## Model Comparison Table

| Model                     | Training Time Complexity | Inference Time Complexity | Interpretability | Robustness to Outliers | Handles Non-linearity | Suitable for Web App Deployment |
|---------------------------|-------------------------|---------------------------|------------------|------------------------|----------------------|--------------------------------|
| **Linear Regression**     | O(n * d^2)             | O(d)                      | High             | Low                    | No                   | Yes (Fast & Lightweight)       |
| **Ridge Regression**      | O(n * d^2)             | O(d)                      | High             | Medium                 | No                   | Yes                            |
| **Lasso Regression**      | O(n * d^2)             | O(d)                      | High             | High                   | No                   | Yes                            |
| **Elastic Net**           | O(n * d^2)             | O(d)                      | High             | High                   | No                   | Yes                            |
| **Decision Tree**         | O(n log n)             | O(log n)                  | Medium           | Low                    | Yes                  | Yes                            |
| **Random Forest**         | O(k * n log n)         | O(k * log n)              | Low              | High                   | Yes                  | Can be heavy                   |
| **Gradient Boosting (XGBoost, LightGBM)** | O(k * n log n) | O(k * log n)   | Low              | High                   | Yes                  | Needs optimization             |
| **Support Vector Regression (SVR)** | O(n^2 * d) (RBF Kernel) | O(n * d) | Low              | Medium                 | Yes                  | Not ideal (Slow inference)     |
| **Neural Networks (MLP)** | O(n * d * l)           | O(d * l)                  | Very Low         | Medium                 | Yes                  | Can be heavy                   |

## Notes:
- n = Number of samples, d = Number of features, k = Number of trees (for ensembles), l = Number of layers (for neural networks)
- **Interpretability:** If understanding feature impact is critical, **Linear Regression, Ridge, Lasso, and Decision Trees** are best.
- **Predictive Power:** If accuracy is the priority, **Random Forest, Gradient Boosting (XGBoost, LightGBM), or Neural Networks** are better.
- **Real-time Inference:** If fast prediction is needed, avoid **SVR** and **Neural Networks**, as they can be slow.

## Model Interpretability Guideline

| Model Type | Global Interpretability | Local Interpretability | Feature Variability |
|------------|------------------------|------------------------|----------------------|
| Linear Regression | Coefficients, PDP, ALE | LIME, SHAP | Handles well, but sensitive to correlated features |
| Decision Trees | Feature Importance, Tree Visualization | Path Analysis | Handles well |
| Random Forest | Permutation Importance, PDP | SHAP, LIME | May mask individual feature importance |
| Gradient Boosting | SHAP, PDP, ALE | SHAP, LIME | Handles well but complex to interpret |
| Neural Networks | SHAP, Surrogate Models | SHAP, Counterfactuals | Harder to interpret |


