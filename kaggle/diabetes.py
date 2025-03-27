import pandas as pd # type: ignore
import numpy as np # type: ignore
df = pd.read_csv("C:\Users\Daivya Kumar Singh\Desktop")
rint(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df['Glucose'])
print()
df[['BloodPressure' , 'Glucose' ]]
print(df.iloc[0])
print(df[df['Age']>10])
from sklearn.preprocessing import StandardScaler # type: ignore
print(df.isnull().sum())
df = df.drop_duplicates()
df['Outcome'].value_counts()
df.groupby('Outcome').mean(numeric_only= True)
X = df.drop(columns = 'Outcome' , axis = 1)
y = df['Outcome']
print(y)
scaler = StandardScaler()
s_d = scaler.transform(X)
print(s_d)
X = s_d    
y = df['Outcome']
from sklearn.model_selection import train_test_split # type: ignore
_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 ,stratify= y, random_state = 2)
from sklearn import svm # type: ignore
classifier = svm.SVC(kernel = 'linear')
classifier.fit(_train, y_train)
from sklearn.metrics import accuracy_score # type: ignore
X_train_prediction = classifier.predict(_train)
training_data_accuracy = accuracy_score(X_train_prediction , y_train)
print("Accuracy Score of the training dataset : " , training_data_accuracy, y_train)