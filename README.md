# Diabetes Prediction using Machine Learning

## Abstract

This project explores the application of machine learning classifiers to predict diabetes, aiming to identify the most accurate and robust approach for early detection and intervention in healthcare management. We evaluate the performance of various individual classifiers, including Decision Trees, Linear Discriminant Analysis (LDA), Naive Bayes, and Logistic Regression, on a comprehensive dataset of health parameters.

## Background

Diabetes mellitus has emerged as a major global health concern, with an escalating prevalence that poses significant challenges to healthcare systems worldwide. Early prediction of diabetes is crucial for timely intervention, enabling healthcare providers to implement preventive measures and improve patient outcomes.

## Machine Learning Models

We implemented and evaluated several machine learning models for diabetes prediction:

- Decision Trees
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Naive Bayes

### Decision Trees

Decision Trees are hierarchical structures that systematically navigate health parameters to classify individuals into diabetic or non-diabetic categories. They offer transparency and interpretability crucial for medical contexts.

Decision Tree Flowchart

## Data Preprocessing

- The dataset underwent cleaning and normalization.
- SMOTE was employed for oversampling, and Random under-sampling was used for the majority class to achieve a balanced dataset of over 20,000 rows.
- The dataset was split into training and validation sets with a 2:1 ratio.

## Evaluation Metrics

We used various evaluation metrics to assess the performance of our models:

- Accuracy
- Sensitivity (Recall)
- Specificity
- ROC-AUC curve
- Precision-Recall curve

Confusion Matrix and Best Classifier Metrics

## Results

The Random Forest (RF) classifier outperformed other models, achieving the highest accuracy of 82.26%. Here's a summary of the key performance metrics:

| Metric    | Random Forest | Decision Tree | KNN   | Logistic Regression | Naive Bayes |
|-----------|---------------|---------------|-------|---------------------|-------------|
| Accuracy  | 82.26%        | 81.54%        | 78.92%| 72.18%              | 70.56%      |
| Precision | 83.47%        | 83.02%        | 80.13%| 73.25%              | 71.89%      |
| Recall    | 80.45%        | 79.73%        | 77.31%| 70.56%              | 67.07%      |
| F1-score  | 82.26%        | 81.54%        | 78.92%| 72.18%              | 70.56%      |

## Conclusions

Our research's primary contribution lies in creating machine learning predictive models for early diabetes detection. The Random Forest classifier achieved the highest accuracy, demonstrating superior performance across all evaluation metrics. This model holds substantial potential to assist medical practitioners in the diabetes diagnosis process.

## Future Work

- Incorporate temporal analysis to explore how the predictive performance of the models evolves over time.
- Investigate the adaptability of models to evolving trends in healthcare data.
- Explore ensemble methods to potentially enhance predictive accuracy further.

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install pandas plotly matplotlib
   ```
3. Run the main script:
   ```
   python final_code.py
   ```

## Contributors

- Rushikesh Gholap
- Apurva Deshpande
- Sushmitha Rajeswari Muppa
- Veda Varshita
