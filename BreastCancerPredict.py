import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import pickle as pickle

## Clean Data
data = pd.read_csv("data/data.csv")
data = data.drop(["Unnamed: 32","id"], axis=1)
data["diagnosis"] = data["diagnosis"].map({"M":1,"B":0})

# Model

### Create Model

X = data.drop(["diagnosis"], axis=1)  # The one we need in order to predict
Y = data["diagnosis"]  # The one we want to predict

### Scale the Data

scaler = StandardScaler()
X = scaler.fit_transform(X)

### Split the Data
x_train, x_test, y_train, y_test = train_test_split(
    X,Y,
    test_size=0.2, # %20 is for testing
    random_state=42 # randomize data for better results
)
### Train Model
model = LogisticRegression()
model.fit(x_train, y_train)

### Test Model
y_pred = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification report: ", classification_report(y_test, y_pred))

### Export Model

with open('model/model.pkl','wb') as file:
  pickle.dump(model, file)
with open('model/scaler.pkl','wb') as file:
  pickle.dump(scaler, file)
