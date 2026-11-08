# Reset the envionment
#%reset -f

# Import Dataset
# import kagglehub

# import os

# dataset_path = kagglehub.dataset_download(
#     "faresashraf1001/supermarket-sales")
  
# df = pd.read_csv(
#     os.path.join(dataset_path, "SuperMarket Analysis.csv"))

import pandas as pd
df = pd.read_csv("midterm_project/market_data.csv")

# No Missing Values
print(df.isnull().sum())

# Check the unique categories of Payment
print(df['Payment'].unique()) 

#Check columns types
print(df.dtypes)

#Split Data Types
category_columns = ["City", "Customer type", "Product line"]
number_columns = ["Sales", "Quantity"]

# Spit X and y
X = df[category_columns + number_columns]
y = df['Payment']

# Train, test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, shuffle =True , random_state = 40)

#Build pipe line
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

preprocessing = ColumnTransformer([
    ("Scaling", StandardScaler(), number_columns),
    ("Encoding",
     OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     category_columns),
])

model = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", LogisticRegression()),
])

# Model train
model.fit(X_train, y_train)
predictions = model.predict(X_test)

## Check Majority Payment
majority_payment = y_train.value_counts().index[0]

baseline_predictions = [majority_payment] * len(y_test)

## Create and Export pickle file
import pickle
with open("midterm_project/payment.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
