# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import and Load Data:
Import necessary libraries (pandas, sklearn) and load the employee churn dataset.

2.Preprocess Data:
Handle missing values, encode categorical variables, and separate features (X) and target (y).

3.Split Data:
Divide the dataset into training and testing sets using train_test_split().

4.Train Model:
Create and train a DecisionTreeClassifier on the training data.

5.Predict and Evaluate:
Use the model to predict churn on test data and evaluate performance using accuracy and confusion matrix.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MOHAMMED PARVEZ S
RegisterNumber:  212223040113
```

```python
import pandas as pd 
import numpy as np
df=pd.read_csv("Employee.csv")
print(df.head())
```
<img width="665" height="292" alt="Screenshot 2025-10-06 104349" src="https://github.com/user-attachments/assets/53d66ee9-ea3b-4fe4-b15c-8b61cdf458fa" />

```python
df.info()
```
<img width="473" height="247" alt="Screenshot 2025-10-06 104438" src="https://github.com/user-attachments/assets/19700967-7004-4138-97e5-03461226d626" />

```python
df.isnull().sum()
```
<img width="242" height="157" alt="Screenshot 2025-10-06 105302" src="https://github.com/user-attachments/assets/a6fe77bf-ded9-4b2b-85b8-258aa14578e9" />

```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()
```
<img width="899" height="152" alt="Screenshot 2025-10-06 105352" src="https://github.com/user-attachments/assets/ec5074f6-ef05-4081-9733-fc5966dd826b" />

```python
df["left"].value_counts()
```
<img width="269" height="61" alt="image" src="https://github.com/user-attachments/assets/4bff609e-5da5-4826-9f67-9691da7dbf66" />

```python
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
<img width="845" height="140" alt="image" src="https://github.com/user-attachments/assets/2fd5ec6b-359e-48f9-a347-554198196944" />

```python
y=df["left"]
y.head()
```
<img width="300" height="94" alt="image" src="https://github.com/user-attachments/assets/db1b35a8-b950-4728-99dc-88e5f727a128" />

```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
y_pred=dt.predict(X_test)
print("Name: MOHAMMED PARVEZ S")
print("RegNo: 212223040113")
print(y_pred)
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,y_pred)
cm=confusion_matrix(Y_test,y_pred)
cr=classification_report(Y_test,y_pred)
print("Accuracy:",accuracy)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(cr)
dt.predict(pd.DataFrame([[0.6,0.9,8,292,6,0,1,2]],columns=["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]))
```
## Output:

<img width="1920" height="1080" alt="Screenshot 2025-10-06 101354" src="https://github.com/user-attachments/assets/0f573360-b603-4c4e-927f-bd0354355ecf" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
