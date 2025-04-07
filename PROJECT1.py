import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
#Creating a sample DataFrame
data={
    'Name':['Alice','Bob','Charlie','David','Eve','Frank','Grace','Helen','Ivy','Jack'],
    'Age':[25,30,35,np.nan,28,33,27,np.nan,31,26],
    'Salary':[5000,7000,np.nan,8000,6000,9000,np.nan,6000,50000,8000],
    'Experience':[3,5,6,np.nan,8,7,4,np.nan,7,60],
    'City':['New york','Los Angeles',np.nan,'Chicago','Miami',np.nan,'London','America','India','Spain']
    }
df=pd.DataFrame(data,columns=['Name','Age','Salary','Experience','City'])
#df.to_csv('PROJECT1.CSV')
df=pd.read_csv('PROJECT1.CSV')
print(df)
print(df.head())
print(df.tail())
print(df.shape)
print(df.size)
print(df.columns)
print(df.dtypes)
print(df.info)
print(df.describe())
print(df.isnull().sum())   #sum of missing values
print(df.isnull().sum().sum())   #total missing values
plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(),cmap="coolwarm",cbar=False)
plt.title("Missing Values-Before Filling")
#plt.show()
df['Age'].fillna(df['Age'].mean(),inplace=True)
print(df)
df['Salary'].fillna(df['Salary'].mean(),inplace=True)
print(df)
df['Experience'].fillna(df['Experience'].mode()[0],inplace=True)
print(df)
df['City'].fillna(df['City'].mode()[0],inplace=True)
print(df)
plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(),cmap="coolwarm",cbar=False)
plt.title("Missing Values-After Filling")
#plt.show()
#encode categorical varibles(label encoding)
label_encoder=LabelEncoder()
df['Name']=label_encoder.fit_transform(df['Name'])
print(df)
df['City']=label_encoder.fit_transform(df['City'])
print(df)
#define features(x)and target(y)
x=df[['Age','Experience']] #independent variable
y=df['Salary']  #target variable
#Normalize
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x)
df[['Age','Salary','Experience']]=scaler.fit_transform(df[['Age','Salary','Experience']])
print("\nFinal Normalized DataFrame:")
print(df)
#Outlier
# Function to detect outliers using IQR method
Q=df['Salary'].quantile([0.25,0.5,0.75])
print(Q)
Q1=df['Salary'].quantile(0.25)
Q3=df['Salary'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outliers=df[(df['Salary']<lower_bound)|(df['Salary']>upper_bound)]
print("\noutliers:")
plt.figure(figsize=(10,6))
sns.boxplot(x=df['Salary'])
plt.title('Boxplot with outliers')
plt.xlabel('Salary')
#plt.show()
w=df['Experience'].quantile([0.25,0.5,0.75])
print(w)
Q1=df['Experience'].quantile(0.25)
Q3=df['Experience'].quantile(0.75)
IQR=Q3-Q1
print(IQR)
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outliers=df[(df['Experience']<lower_bound)|(df['Experience']>upper_bound)]
print("\noutliers:")
plt.figure(figsize=(10,6))
sns.boxplot(x=df['Experience'])
plt.title('Boxplot with outliers')
plt.xlabel('Experience')
#plt.show()
#split data into train and test
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,random_state=42)
knn=KNeighborsClassifier(n_neighbors=3)
log_reg=LogisticRegression()
svm_model=SVC()
#Train models on training data
knn.fit(x_train,y_train)
log_reg.fit(x_train,y_train)
svm_model.fit(x_train,y_train)
#Make prediction on testing data
y_pred_knn=knn.predict(x_test)
y_pred_log=log_reg.predict(x_test)
y_pred_svm=svm_model.predict(x_test)
#Evaluate model performance (Accuracy and metrics)
knn_accuracy=accuracy_score(y_test,y_pred_knn)
log_reg_accuracy=accuracy_score(y_test,y_pred_log)
svm_accuracy=accuracy_score(y_test,y_pred_svm)
#Display performance results
performance_matrix=pd.DataFrame({
    'Model':['KNN','Logistic Regression','SVM'],
    'Accuracy':[knn_accuracy,log_reg_accuracy,svm_accuracy],
    })
print(performance_matrix)
