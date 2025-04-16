# Exploratory Data Analysis (EDA) for MMA Marketing Data Sample

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot aesthetics
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(style='whitegrid')

# Load the dataset
df = pd.read_csv('MMA marketing_data_sample.csv')
#replace the terme 'k' by 'education' for clarity
df.rename(columns={'k': 'education'}, inplace=True)
#Missing values are stated as 'unknown' in the dataset:
df.replace('unknown', np.nan, inplace=True)

# Basic Data Inspection
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics for Numerical Variables:")
print(df.describe())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# -----------------------------
# Visualisation Section
# -----------------------------

# # 1. Distribution of Age
plt.figure()
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Count Plot for Job (categorical)
plt.figure()
sns.countplot(x='job', data=df, order=df['job'].value_counts().index)
plt.title('Job Distribution')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# # 3. Count Plot for Marital Status
plt.figure()
sns.countplot(x='marital', data=df, order=df['marital'].value_counts().index)
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()

# 4. Count Plot for Education Level
plt.figure()
sns.countplot(x='education', data=df, order=df['education'].value_counts().index)
plt.title('Education Level Distribution')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 5. Boxplot of Age by Target Variable (y)
plt.figure()
sns.boxplot(x='y', y='age', data=df)
plt.title('Age Distribution by Deposit Acceptance (y)')
plt.xlabel('Deposit Acceptance (y)')
plt.ylabel('Age')
plt.show()

# 6. Count Plot for Contact Type
plt.figure()
sns.countplot(x='contact', data=df, order=df['contact'].value_counts().index)
plt.title('Contact Type Distribution')
plt.xlabel('Contact Type')
plt.ylabel('Count')
plt.show()

# 7. Count Plot for Previous Campaign Outcome (poutcome)
plt.figure()
sns.countplot(x='poutcome', data=df, order=df['poutcome'].value_counts().index)
plt.title('Previous Campaign Outcome Distribution')
plt.xlabel('poutcome')
plt.ylabel('Count')
plt.show()

# 8. Distribution of Campaign Contacts and Previous Contacts
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df['campaign'], bins=20, kde=False, ax=ax[0])
ax[0].set_title('Distribution of Campaign Contacts')
ax[0].set_xlabel('Campaign Contacts')

sns.histplot(df['previous'], bins=20, kde=False, ax=ax[1])
ax[1].set_title('Distribution of Previous Contacts')
ax[1].set_xlabel('Previous Contacts')

plt.tight_layout()
plt.show()

# 9. Correlation Matrix for Numerical Variables
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix for Numerical Variables')
plt.show()

# 10. Heatmap for Missing Data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values')
plt.show()


# -----------------------------
# Preprocessing Section
# -----------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Copy the cleaned DataFrame from Part 1
df_model = df.copy()

# Rename column 'k' to 'education' for clarity
df_model.rename(columns={'k': 'education'}, inplace=True)

# List of categorical columns
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
            'contact', 'month', 'day_of_week', 'poutcome']

# For categorical features, fill missing values with a new category 'missing'
for col in cat_cols:
    df_model[col] = df_model[col].fillna('missing')

# List of numerical columns
num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 
            'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
            'euribor3m', 'nr.employed']

# For numerical features, fill missing values with the median
for col in num_cols:
    df_model[col] = df_model[col].fillna(df_model[col].median())

# Convert target variable y: 'yes' to 1 and 'no' to 0
df_model['y'] = df_model['y'].map({'yes': 1, 'no': 0})

# One-hot encode categorical variables (dropping the first level to avoid multicollinearity)
df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

# Define features and target
X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

# Split into training and test sets (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features using StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])





# -----------------------------
# Model Building and Evaluation Section
# -----------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ---- Logistic Regression ----
model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]

print("Logistic Regression Performance:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("AUC Score:", roc_auc_score(y_test, y_prob_lr))
print("\n")

# ---- Decision Tree ----
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
y_prob_dt = model_dt.predict_proba(X_test)[:, 1]

print("Decision Tree Performance:")
print(confusion_matrix(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
print("AUC Score:", roc_auc_score(y_test, y_prob_dt))
print("\n")

# ---- Random Forest ----
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]

print("Random Forest Performance:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("AUC Score:", roc_auc_score(y_test, y_prob_rf))
print("\n")

# ---- Plot ROC Curves for All Models ----
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, y_prob_lr):.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_score(y_test, y_prob_dt):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Classifiers')
plt.legend()
plt.show()
