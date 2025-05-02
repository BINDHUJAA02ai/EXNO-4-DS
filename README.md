# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![437555175-076120af-c445-4cc8-9785-1244e570c8c0](https://github.com/user-attachments/assets/d518ffa5-0b45-4009-bab0-281bd9033f4c)
```
data.isnull().sum()
```
![437555302-eba26a9e-af61-4f04-92f8-3a2943d61f1e](https://github.com/user-attachments/assets/e82230f9-9bc8-40c4-9512-4fde1b4115b5)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![437555506-50b73888-5cd3-4b4e-ba13-5c07cf4cdeb7](https://github.com/user-attachments/assets/d54c5c85-2ab3-44ad-be46-96a32f8031d5)
```
data2=data.dropna(axis=0)
data2
```
![437555692-5752d95c-5ee4-43ba-9609-2f077a3eb10f](https://github.com/user-attachments/assets/2a4e270d-02d7-426a-9db8-d1d9b3f8e12c)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![437555837-0bb2da6e-ba8c-413f-b3d4-9c92795df540](https://github.com/user-attachments/assets/dadc2b44-c08f-49a9-b6b2-02f9ccfee22c)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![437555958-490d6a1d-68ed-4605-8fad-e77c891adfa9](https://github.com/user-attachments/assets/22e4cb48-c772-43e6-8ef1-97ccae3a7ebb)
```
data2
```
![437556083-2df2c662-a4cd-4c7e-a520-f967fc0c0309](https://github.com/user-attachments/assets/771d393b-38aa-4fee-b512-07b05537faad)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![437556262-2ffa8851-3018-4f99-bf55-a57e4079f282](https://github.com/user-attachments/assets/7be490b6-6411-4de0-92af-20b5cc867b75)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![437556378-e9fd21ec-355d-458e-ac3a-d5716abe7daa](https://github.com/user-attachments/assets/5606c816-756c-4cf8-bef7-a1d8a8d504b8)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![437556542-5f441028-4d94-41ba-a5b6-e73e8311cbf1](https://github.com/user-attachments/assets/bd494648-47d2-4794-a4b3-59cebe2e61de)
```
y=new_data['SalStat'].values
print(y)
```

![437556657-59250349-0117-42a3-9cc8-b98be6521e34](https://github.com/user-attachments/assets/55ce46a9-0d0b-4b46-a662-b6c17e80a9dc)
```
x=new_data[features].values
print(x)

```
![437556834-06c9ac7a-6926-4fb4-bd70-7a9545e810c0](https://github.com/user-attachments/assets/bda36051-9053-4370-a502-b2aea3694eaf)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![437557037-2543f919-786f-4799-a7b1-9f92b631ff11](https://github.com/user-attachments/assets/79ecc0c5-bf5c-4b7f-92f1-b021843b221d)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![437557106-44322935-d116-497c-a9f6-1aacaeb9b022](https://github.com/user-attachments/assets/c6d6781e-78cc-48b4-b9bd-2ce1e29d1e9e)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![437557282-fb4bfcd3-7f81-41bd-90cf-8ee1004704cd](https://github.com/user-attachments/assets/80289f07-4ffc-46ce-ab43-926d370efb27)
```



 print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
 ![437557388-fc077826-0c7e-497d-ae82-039aba1457ea](https://github.com/user-attachments/assets/d770da7f-d6b5-4b61-ba0d-61fb3922a418)
 ```
data.shape
```
![437557496-1fe044eb-fd93-473f-bfbf-3bb9658a1c5d](https://github.com/user-attachments/assets/99be7aaf-6814-4916-b0e0-416d762af081)
```

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

![437557651-35254ec6-12e6-489f-b911-ad18cd74c352](https://github.com/user-attachments/assets/fb2c2d12-0797-4a07-b5c1-a8e9e91e91a1)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![437557870-adb8b4fa-5f85-485e-b4c0-e9d6bb9c8870](https://github.com/user-attachments/assets/1cedec99-321a-47ec-8109-1aecffdf47df)
```
tips.time.unique()
```
![322338497-f77bc757-8a31-4a5d-be15-5a447e6549c6](https://github.com/user-attachments/assets/d924b474-00e6-4901-91a9-592f57d75158)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![437558086-8dbd3f42-d282-4e30-8a6e-d98f25471203](https://github.com/user-attachments/assets/f2c477ea-5f87-4c51-8fcc-7b11a3a7a320)


```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![437558188-18ef6303-8d5d-43e2-808f-b44a2cc05148](https://github.com/user-attachments/assets/9517dd80-4b7c-4977-9631-866fb938ad1e)


# RESULT:
    Thus, Feature selection and Feature scaling has been used on thegiven dataset.
