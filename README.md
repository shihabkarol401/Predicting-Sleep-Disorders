# Predicting Sleep Disorders
This project focuses on predicting sleep disorders using patient health, lifestyle, and demographic data. It integrates data preprocessing, exploratory analysis, feature engineering, and advanced machine learning techniques to build a robust predictive model. After evaluating multiple classifiers, a Random Forest (with selected features) model emerged as the best performer, providing strong accuracy, balanced precision–recall, and excellent ROC-AUC performance for identifying individuals at risk of sleep disorders.

# Project Workflow
1. Data Preparation: Loaded the Sleep_health_and_lifestyle.csv dataset, inspected data types and missing values, and imputed all missing entries in Sleep Disorder using grouped mode (by BMI Category, Occupation, and Stress Level) followed by global mode to ensure a complete dataset.
2. Exploratory Data Analysis (EDA): Analyzed numerical feature distributions (Age, Sleep Duration, Physical Activity, Stress Level, Heart Rate, Daily Steps) using visualizations and boxplots across sleep disorder categories; examined categorical relationships (Gender, Age groups, Occupation, BMI Category, Blood Pressure, Heart Rate); identified outliers via the IQR method with Winsorization applied to Heart Rate; and generated a correlation heatmap to assess inter-feature relationships.
3. Feature Engineering: Applied one-hot encoding to categorical variables (Gender, Occupation, BMI Category, Sleep Disorder), transformed Blood Pressure into Systolic and Diastolic Pressure, and dropped the original composite feature after transformation.
4. Data Scaling: Scaled numerical variables using StandardScaler to ensure uniform feature ranges prior to training.
5. Handling Class Imbalance: Addressed imbalance in the target class Sleep Disorder_Sleep Apnea (68.18% False, 31.82% True) by applying SMOTE to the training dataset, ensuring balanced class representation and reducing model bias.

# Model Training
Multiple classification algorithms were trained and evaluated:
- Logistic Regression
- Gaussian Naïve Bayes
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVC)
- Random Forest
- XGBoost
- Gradient Boosting

# Model Evaluation
Models were compared using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Initial evaluation showed that tree-based ensemble moedls outperformed others, with Random Forest achieving the highest ROC-AUC (0.9551) and Decision Tree delivering strong F1-Score performance.

# Feature Selection
Extracted feature importance from the Random Forest model to identify the top 13 features, retrained all models using only these selected features, and observed stable overall performance with Random Forest (Selected Features) achieving an improved ROC-AUC of 0.9575 while maintaining a high F1-Score.

# Deployment
Access the deployed model:

https://shihabkarol-sleep-disorder-prediction.hf.space/?logs=container&__theme=system&deep_link=QxI6-bOoEeQ

# Key Insights
Analysis revealed that age and occupation significantly influence sleep disorder prevalence, physical activity and stress levels are major lifestyle contributors, cardiovascular metrics such as BMI category, blood pressure, and heart rate are strong predictors, and sleep duration and quality are critical determinants of classification accuracy.

# Future Scope
Extend the project by exploring deep learning models (MLP, RNN) and ensemble stacking, integrating Explainable AI techniques (SHAP, LIME) for interpretability, enhancing feature engineering with interaction terms, temporal trends, and domain-specific indicators, and conducting clinical validation using real-world or longitudinal datasets.
