# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the `pandas` library and read the dataset `Placement_Data.csv` into a DataFrame `data`.
2. Display the first few rows of `data` using `data.head()` to check its contents.
3. Create a copy of `data` named `data1` to work on.
4. Drop the columns `"sl_no"` and `"salary"` from `data1` using `data1.drop()`.
5. Display the first few rows of `data1` after dropping columns to confirm changes.
6. Check for missing values in `data1` using `data1.isnull().sum()`.
7. Check for duplicate rows in `data1` using `data1.duplicated().sum()`.
8. Import `LabelEncoder` from `sklearn.preprocessing` for encoding categorical variables.
9. Create an instance of `LabelEncoder` named `le`.
10. Encode categorical columns (`"gender"`, `"ssc_b"`, `"hsc_b"`, `"hsc_s"`, `"degree_t"`, `"workex"`, `"specialisation"`, `"status"`) in `data1` using `le.fit_transform()`.
11. Display `data1` after encoding to check that categorical columns are encoded into numerical values.
12. Extract the independent variables `x` (all columns except `"status"`) from `data1`.
13. Extract the dependent variable `y` (the `"status"` column) from `data1`.
14. Split `x` and `y` into training and testing sets using `train_test_split()` with 80% training and 20% testing data, setting `random_state` to 0 for reproducibility.
15. Import `LogisticRegression` from `sklearn.linear_model`.
16. Create an instance of `LogisticRegression` named `lr` with the solver set to `"liblinear"`.
17. Train the model on the training data using `lr.fit()` with `x_train` and `y_train`.
18. Make predictions on the test set `x_test` using `lr.predict()` and store them in `y_pred`.
19. Print or return `y_pred` to display the predicted output.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Tanushree
RegisterNumber:212223100057  
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
X=data1.iloc[:,: -1]
X
Y=data1["status"]
Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression (solver="liblinear")
lr.fit(X_train,Y_train)
Y_pred= lr.predict(X_test)
Y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,Y_pred)
accuracy
```

## Output:

## predictions
![image](https://github.com/user-attachments/assets/013b160f-6694-4c79-b7bc-585ba5c6a95c)

## accuracy score
![image](https://github.com/user-attachments/assets/ab294639-d45d-4b1b-b1f6-13c94f9ee2f8)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
