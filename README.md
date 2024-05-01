# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: A.ARUVI
RegisterNumber:  212222230014
*/

import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head() 
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_predict)
mse
r2=metrics.r2_score(y_test,y_predict)
r2
dt.predict([[6,7]])
```
## Output:
### Initial Dataset:

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120443233/3fe14b5f-7427-42fc-99ae-4d12f918c900)

### data.info():

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120443233/1090ab1f-8216-4ec9-918b-a21d804530c4)

### Optimisation of Null values:
![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120443233/4b8db974-25b4-4c29-aad9-4f3f1f090b29)

### Converting string literals to numerical values using label encoder:

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120443233/306e8b44-9eed-4e50-8c66-5475861416f0)

### Mean Squared Error:

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120443233/13165494-2dd5-4334-b988-d85b987317c5)

### R2 (variance):

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120443233/f373bb3c-0e54-4d1e-826e-35d3054f5c6a)

### Data prediction:

![image](https://github.com/Anandanaruvi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/120443233/a7ab165d-00d7-424e-ab2f-2371b5b84375)


### Result:

Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
