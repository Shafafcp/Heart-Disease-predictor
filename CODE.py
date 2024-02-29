# Heart-Disease-predictor
Heart disease predictor
Suppose you are appointed as a Data scientist in any Pharma Company. That company makes medicine for heart disease. Your senior manager has given several clinical parameters about a patient, can you predict whether or not the patient has heart disease?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#Load the dataset
data=pd.read_csv(r"C:\Users\shafa\Downloads\heart.csv")
Explore the data

data.head()
age	sex	cp	trtbps	chol	fbs	restecg	thalachh	exng	oldpeak	slp	caa	thall	output
0	63	1	3	145	233	1	0	150	0	2.3	0	0	1	1
1	37	1	2	130	250	0	1	187	0	3.5	0	0	2	1
2	41	0	1	130	204	0	0	172	0	1.4	2	0	2	1
3	56	1	1	120	236	0	1	178	0	0.8	2	0	2	1
4	57	0	0	120	354	0	1	163	1	0.6	2	0	2	1
data.tail()
age	sex	cp	trtbps	chol	fbs	restecg	thalachh	exng	oldpeak	slp	caa	thall	output
298	57	0	0	140	241	0	1	123	1	0.2	1	0	3	0
299	45	1	3	110	264	0	1	132	0	1.2	1	0	3	0
300	68	1	0	144	193	1	1	141	0	3.4	1	2	3	0
301	57	1	0	130	131	0	1	115	1	1.2	1	1	3	0
302	57	0	1	130	236	0	0	174	0	0.0	1	1	2	0
data.isnull().sum()
age         0
sex         0
cp          0
trtbps      0
chol        0
fbs         0
restecg     0
thalachh    0
exng        0
oldpeak     0
slp         0
caa         0
thall       0
output      0
dtype: int64
data.describe()
age	sex	cp	trtbps	chol	fbs	restecg	thalachh	exng	oldpeak	slp	caa	thall	output
count	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000	303.000000
mean	54.366337	0.683168	0.966997	131.623762	246.264026	0.148515	0.528053	149.646865	0.326733	1.039604	1.399340	0.729373	2.313531	0.544554
std	9.082101	0.466011	1.032052	17.538143	51.830751	0.356198	0.525860	22.905161	0.469794	1.161075	0.616226	1.022606	0.612277	0.498835
min	29.000000	0.000000	0.000000	94.000000	126.000000	0.000000	0.000000	71.000000	0.000000	0.000000	0.000000	0.000000	0.000000	0.000000
25%	47.500000	0.000000	0.000000	120.000000	211.000000	0.000000	0.000000	133.500000	0.000000	0.000000	1.000000	0.000000	2.000000	0.000000
50%	55.000000	1.000000	1.000000	130.000000	240.000000	0.000000	1.000000	153.000000	0.000000	0.800000	1.000000	0.000000	2.000000	1.000000
75%	61.000000	1.000000	2.000000	140.000000	274.500000	0.000000	1.000000	166.000000	1.000000	1.600000	2.000000	1.000000	3.000000	1.000000
max	77.000000	1.000000	3.000000	200.000000	564.000000	1.000000	2.000000	202.000000	1.000000	6.200000	2.000000	4.000000	3.000000	1.000000
Data Preprocessing

#Handling missing values(if any)
data=data.dropna()
#Splitting the data into features and target
X=data.drop('output',axis=1)
y=data['output']
​
#Seperating into training and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
​
#Data scaling 
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
​
Apply Machine Learning algorithms
a.Logistic Regeression

#Create a logisitic egression model
logistic=LogisticRegression()
logistic.fit(X_train,y_train)
​
#Make predicitons on test data
pred_logistic=logistic.predict(X_test)
​
#Evaluate model
accuracy_logistic=accuracy_score(y_test,pred_logistic)
print(f"Logistic Regression Accuracy : {accuracy_logistic}")
print("Confusion Matrix : \n",confusion_matrix(y_test,pred_logistic))
print("Classification Report \n",classification_report(y_test,pred_logistic))
Logistic Regression Accuracy : 0.8524590163934426
Confusion Matrix : 
 [[25  4]
 [ 5 27]]
Classification Report 
               precision    recall  f1-score   support

           0       0.83      0.86      0.85        29
           1       0.87      0.84      0.86        32

    accuracy                           0.85        61
   macro avg       0.85      0.85      0.85        61
weighted avg       0.85      0.85      0.85        61

b. K-Nearest neighbors Classifier

#Create K_Nearest neighbor model
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
​
#prediciton
pred_knn=knn.predict(X_test)
​
#Evaluate
accuracy_knn=accuracy_score(y_test,pred_knn)
​
print(f"KNN Accuracy : {accuracy_knn}")
print("Confusion Matrix : \n",confusion_matrix(y_test,pred_knn))
print("Classification Report \n",classification_report(y_test,pred_knn))
KNN Accuracy : 0.9016393442622951
Confusion Matrix : 
 [[27  2]
 [ 4 28]]
Classification Report 
               precision    recall  f1-score   support

           0       0.87      0.93      0.90        29
           1       0.93      0.88      0.90        32

    accuracy                           0.90        61
   macro avg       0.90      0.90      0.90        61
weighted avg       0.90      0.90      0.90        61

c.Support Vector Machine

#Create SVM model
svm=SVC()
svm.fit(X_train,y_train)
​
#predicition
pred_svm=svm.predict(X_test)
​
#Evaluate
accuracy_svm=accuracy_score(y_test,pred_svm)
​
print(f"SVM Accuracy : {accuracy_svm}")
print("Confusion Matrix : \n",confusion_matrix(y_test,pred_svm))
print("Classification Report \n",classification_report(y_test,pred_svm))
SVM Accuracy : 0.8688524590163934
Confusion Matrix : 
 [[26  3]
 [ 5 27]]
Classification Report 
               precision    recall  f1-score   support

           0       0.84      0.90      0.87        29
           1       0.90      0.84      0.87        32

    accuracy                           0.87        61
   macro avg       0.87      0.87      0.87        61
weighted avg       0.87      0.87      0.87        61

d. Decision Tree Classifier

#Create DT model
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
​
#predict
pred_dt=dt.predict(X_test)
​
#Evaluate
accuracy_dt=accuracy_score(y_test,pred_dt)
​
print(f"Decision tree Accuracy : {accuracy_dt}")
print("Confusion Matrix : \n",confusion_matrix(y_test,pred_dt))
print("Classification Report \n",classification_report(y_test,pred_dt))
Decision tree Accuracy : 0.819672131147541
Confusion Matrix : 
 [[26  3]
 [ 8 24]]
Classification Report 
               precision    recall  f1-score   support

           0       0.76      0.90      0.83        29
           1       0.89      0.75      0.81        32

    accuracy                           0.82        61
   macro avg       0.83      0.82      0.82        61
weighted avg       0.83      0.82      0.82        61

e. Random Forest Classifier

#create model
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
​
#prediciton
pred_rf=rf.predict(X_test)
​
#Evaluate
accuracy_rf=accuracy_score(y_test,pred_rf)
​
print(f"RandomForest Accuracy : {accuracy_rf}")
print("Confusion Matrix : \n",confusion_matrix(y_test,pred_rf))
print("Classification Report \n",classification_report(y_test,pred_rf))
RandomForest Accuracy : 0.8524590163934426
Confusion Matrix : 
 [[24  5]
 [ 4 28]]
Classification Report 
               precision    recall  f1-score   support

           0       0.86      0.83      0.84        29
           1       0.85      0.88      0.86        32

    accuracy                           0.85        61
   macro avg       0.85      0.85      0.85        61
weighted avg       0.85      0.85      0.85        61

# Example patient data
new_patient_data = np.array([55, 1, 0, 140, 240, 0, 1, 170, 0, 2.5, 1, 0, 2]).reshape(1, -1)
# Standardize the new patient data
new_patient_data_standardized = scaler.transform(new_patient_data)
# Make predictions for the new patient
prediction = knn.predict(new_patient_data_standardized)
​
​
if prediction[0] == 1:
    print("The model predicts that the patient has a chance of heart disease.")
else:
    print("The model predicts that the patient does not have a chance of heart disease.")
​
The model predicts that the patient has a chance of heart disease.
