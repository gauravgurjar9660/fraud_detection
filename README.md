## 🏆 Model Performance Comparison

To ensure the highest accuracy for fraud detection, I experimented with various boosting algorithms. The performance was quite consistent across the models, showing robust feature extraction:

| Model | PR-AUC Score |
| :--- | :--- |
| Random Forest | 76.10% |
| LightGBM | 76.25% |
| **XGBoost (Best)** | **76.35%** |

*Note: While Ensemble methods were tested, XGBoost provided the best balance of performance and computational efficiency for this dataset.*

## 🧠 Technical Learning Journey

* **Handling Class Imbalance with SMOTE:** Financial datasets are typically highly skewed (majority of transactions are genuine). I utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class (fraud). This prevented the model from being biased towards the majority class and significantly improved the detection of fraudulent patterns.
  
* **Model Selection & Ensemble Strategy:** I compared Random Forest, LightGBM, and XGBoost. While all models performed consistently, **XGBoost** was selected for its superior PR-AUC score and stability in handling sparse, imbalanced features. I explored ensemble techniques to ensure the model captured diverse transaction behaviors.

* **Why PR-AUC Over Accuracy?** In fraud detection, standard accuracy is misleading. I focused on **Precision-Recall AUC**, which provides a better measure of the model's ability to identify the minority (fraud) class without being overwhelmed by the high volume of genuine transactions.

* **Performance Insights:** The model achieves **88% Recall**, ensuring the vast majority of fraud cases are caught, balanced with **60% Precision** to minimize false alerts. This trade-off is optimized for a production-level financial security system.
