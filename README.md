📊 Telecom Customer Churn Prediction
📝 Project Overview
In the highly competitive telecommunications industry, retaining customers is often more cost-effective than acquiring new ones. This project aims to build a machine learning model to predict Customer Churn (whether a customer will leave the service).

By identifying at-risk customers early, telecom companies can take proactive measures (e.g., special offers, better support) to improve retention rates.

#🎯 Objectives
Analyze customer data to find patterns associated with churn.

Preprocess the data (handling missing values, encoding categorical variables, scaling).

Compare multiple machine learning algorithms (SVM, Random Forest, XGBoost, Logistic Regression).

Optimize the best model using Hyperparameter Tuning.

Address Class Imbalance using SMOTE.

Evaluate the model using metrics like Recall, Precision, and ROC-AUC.


#🛠️ Tools & Technologies
Programming Language: Python

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-Learn, XGBoost

Imbalanced Data Handling: SMOTE (Imbalanced-Learn)

Model Saving: Joblib

#🚀 Key Steps
Data Cleaning:

Dropped irrelevant columns (customerID).

Handled missing values in TotalCharges.

Converted the target variable Churn to binary (0/1).

Feature Engineering:

Applied One-Hot Encoding for categorical features (e.g., Payment Method, Internet Service).

Applied Standard Scaling to numerical features to normalize the data range.

Model Selection:

Used GridSearchCV to compare SVM, Random Forest, Logistic Regression, and XGBoost.

Winner: XGBoost achieved the highest accuracy (~80%).

Handling Imbalance:

The original dataset was imbalanced (fewer churners than non-churners).

Applied SMOTE to oversample the minority class, significantly improving the model's ability to detect churners (Recall increased from 0.49 to 0.69).

#📈 Results & Evaluation
Model Performance (After SMOTE)
Algorithm: XGBoost Classifier

Accuracy: 77%

Recall (Churn Class): 0.69 (Good ability to catch potential churners)

ROC-AUC Score: 0.83

Interpretability
ROC Curve: The model shows a strong separation capability with an AUC of 0.83.

Feature Importance: The analysis revealed that Contract Type, Monthly Charges, and Tenure are the top drivers for customer churn. Short-term contracts and high monthly bills significantly increase churn risk.

#💻 How to Run
Clone the repository:

#Bash

git clone https://github.com/your-username/Telecom-Churn-Prediction.git
cd Telecom-Churn-Prediction
Install dependencies:

#Bash

pip install -r requirements.txt
Run the Notebook: Open notebooks/Churn_Analysis.ipynb in Jupyter Notebook or VS Code to see the full analysis.

Use the Model (Example):

#Python

import joblib

model = joblib.load('models/churn_prediction_model.pkl')


#👤 Author

Ali Abubakr Farag Ali

Artificial Intelligence & Data Science Student

http://linkedin.com/in/aliabubakr1

Feel free to star ⭐ this repository if you find it helpful!
