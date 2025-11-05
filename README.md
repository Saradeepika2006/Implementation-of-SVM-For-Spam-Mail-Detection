# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.

2.Read the dataset and separate the independent and dependent variables.

3.Split the dataset into training and testing.

4.Do preprocessing if needed, in this case vectorization is needed which is done using CountVectorizer()

5.Train the model using SVC() algorithm and .fit()

6.Predict the model on x_test.

7.Measure its accuracy

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Mopuri Saradeepika
RegisterNumber:  212224040201
*/

import pandas as pd
    data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
    data.info()
    
    x=data['v2'].values
    y=data['v1'].values
    x.shape
    y.shape
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer()
    x_train=cv.fit_transform(x_train)
    x_test=cv.transform(x_test)
    x_train
    
    from sklearn.svm import SVC
    svc=SVC()
    svc.fit(x_train,y_train)
    y_pred=svc.predict(x_test)
    y_pred
    
    from sklearn.metrics import accuracy_score
    acc=accuracy_score(y_test,y_pred)
    acc

```

## Output:

<img width="1199" height="379" alt="Screenshot 2025-11-05 154002" src="https://github.com/user-attachments/assets/1d0f6ef9-9ec1-4e8e-9e0f-332c970e3e85" />

<img width="1119" height="26" alt="Screenshot 2025-11-05 154022" src="https://github.com/user-attachments/assets/043031e2-e514-4b80-99fb-281ff3c054be" />

<img width="1160" height="41" alt="Screenshot 2025-11-05 154039" src="https://github.com/user-attachments/assets/0adc8fa1-01b3-4008-834f-201361e8bd97" />

<img width="1195" height="38" alt="Screenshot 2025-11-05 154052" src="https://github.com/user-attachments/assets/9b163dc2-16ac-4faf-9427-871e8eb88cb6" />

<img width="803" height="43" alt="Screenshot 2025-11-05 154101" src="https://github.com/user-attachments/assets/57bfa867-09f2-45c8-b5d1-bdad444217fd" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
