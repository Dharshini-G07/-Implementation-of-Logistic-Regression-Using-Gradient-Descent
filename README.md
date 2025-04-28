# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## AlgorithmLoad the Dataset


1.Load dataset: dataset = pd.read_csv('Placement_Data.csv')

2.Preprocess data: Drop irrelevant columns sl_no and salary.

3.Convert categorical variables: Convert columns like gender, ssc_b, hsc_b, degree_t, workex, specialisation, status, hsc_s to numeric codes using astype('category').cat.codes.

4.Separate features and target: X = dataset.iloc[:, :-1].values, Y = dataset.iloc[:, -1].values.

5.Initialize weights: theta = np.random.randn(X.shape[1]).

6.Implement sigmoid function: def sigmoid(z): return 1 / (1 + np.exp(-z)).

7.Define loss function: def loss(theta, X, y): return -np.sum(y * np.log(sigmoid(X.dot(theta))) + (1 - y) * np.log(1 - sigmoid(X.dot(theta)))).

8.Implement gradient descent: def gradient_descent(theta, X, y, alpha, num_iterations): theta -= alpha * X.T.dot(sigmoid(X.dot(theta)) - y) / len(y).

9.Make predictions: y_pred = np.where(sigmoid(X.dot(theta)) >= 0.5, 1, 0).

10.Evaluate model: accuracy = np.mean(y_pred.flatten() == y).







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
dataset=dataset.drop('sl_no',axis=1)
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
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
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
print(Y)
xnew= np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew= np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

```

## Output:

## Dataset
![image](https://github.com/user-attachments/assets/888d7e62-a3ca-4bc7-8164-86e9f32a6af4)
## dtypes
![image](https://github.com/user-attachments/assets/df7943f5-8ba8-41ae-9fcd-8069e0b4ad15)
## dataset
![image](https://github.com/user-attachments/assets/316affd3-4f4b-48f7-b35b-3d4ab627d0ac)
## y array
![image](https://github.com/user-attachments/assets/7853ca60-aa1d-495b-89f7-e66c9fd16aab)
## Accuracy
![image](https://github.com/user-attachments/assets/26aaa5ee-29b8-43b5-b74e-6bfebffec65e)
## y_pred
![image](https://github.com/user-attachments/assets/169eca79-c656-4e6d-a37b-a17b29cea12f)
## y
![image](https://github.com/user-attachments/assets/46cc78b2-ae55-4457-8117-a0e2201f6f79)
## y_prednew
![image](https://github.com/user-attachments/assets/00ac94c8-150b-4f2b-9ac0-3759a24babfc)
## y_prednew
![image](https://github.com/user-attachments/assets/ec1975d4-3442-4cfe-acb3-1573836b6624)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

