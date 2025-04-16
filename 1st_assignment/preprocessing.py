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
