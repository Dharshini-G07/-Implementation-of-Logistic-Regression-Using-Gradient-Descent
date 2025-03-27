# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Import the required libraries.
2.  Load the dataset.
3.  Define X and Y array.
4.  Define a function for costFunction,cost and gradient.
5.  Define a function to plot the decision boundary. 6.Define a function to predict the 
    Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Priyadharshini G
RegisterNumber:  212224230209
*/
```
```
import pandas as pd
import numpy as np
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset= dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
X= dataset.iloc[:, :-1].values
Y= dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred = np.where(h>= 0.5,1,0)
    return y_pred
y_pred = predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```

## Output:

![image](https://github.com/user-attachments/assets/5d7a1f64-ed88-47e1-8bb6-eeea2997bc4d)
![image](https://github.com/user-attachments/assets/3ec353b2-11ad-4399-b3e6-ec8643dba945)
![image](https://github.com/user-attachments/assets/b5edb8c0-8c05-4186-ae54-c3f8cad8a44a)
![image](https://github.com/user-attachments/assets/be475ab9-f717-4e88-8acd-b299993391ab)
![image](https://github.com/user-attachments/assets/a5de55fe-9886-4676-a656-92738cec249b)
![image](https://github.com/user-attachments/assets/d674cf8e-f626-45e1-a0f6-13103f21a333)
![image](https://github.com/user-attachments/assets/51b1e8ef-b0ab-437e-8dfc-6c428f4ddc2e)
![image](https://github.com/user-attachments/assets/fb30cf70-fb76-4e43-a462-429ae05bf271)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

