# About data
All data from Kaggle
It contains information about various features extracted from breast cancer samples. The following preprocessing steps are applied to the data:

Removal of unnecessary columns: The columns "Unnamed: 32" and "id" are dropped from the dataset as they are not relevant for the prediction task.
Conversion of the target variable: The "diagnosis" column, which represents the diagnosis of the breast cancer samples, is converted from categorical labels ("M" for malignant and "B" for benign) to numeric values (1 for malignant and 0 for benign)

# Model creation
The model creation process involves the following steps:

* Splitting the data into input features (X) and the target variable (Y): The input features are obtained by dropping the "diagnosis" column from the dataset, while the target variable is set as the "diagnosis" column.
* Scaling the data: The input features (X) are standardized using the StandardScaler from sklearn.preprocessing to ensure that all features have the same scale.
* Splitting the data into training and testing sets: The standardized input features (X) and the target variable (Y) are split into training and testing sets using the train_test_split function from sklearn.model_selection. The testing set size is set to 20% of the total data, and a random state of 42 is used for reproducibility.
* Training the model: A logistic regression model is created using the LogisticRegression class from sklearn.linear_model, and it is trained on the training data.

# Model Evaluation
After training the model, it is evaluated using the testing set. The following evaluation metrics are computed:

* Accuracy: The accuracy score is calculated by comparing the predicted values (y_pred) with the actual target values (y_test).
* Classification report: A detailed classification report is generated using the classification_report function from sklearn.metrics. It includes precision, recall, F1-score, and support for both the malignant and benign classes.
  
The evaluation metrics are printed to the console using the print function.

# Model Export
Finally, the trained model and the scaler used for feature scaling are saved for future use. The model is saved as a pickle file named model.pkl, and the scaler is saved as scaler.pkl in the model directory.

Please note that this code assumes that the necessary dependencies, such as pandas and scikit-learn, are already installed in your environment.
