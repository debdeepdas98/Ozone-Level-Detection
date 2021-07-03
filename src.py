import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
eighthr = pd.read_csv("G:\FILES\Resources\DataSets\Ozone Level
Detection\eighthr.data", header = -1)
columns_eighthr =
['Date','WSR0','WSR1','WSR2','WSR3','WSR4','WSR5','WSR6','WSR7','WSR8','WSR
9','WSR10','WSR11','WSR12','WSR13','WSR14','WSR15','WSR16','WSR17','WSR18'
,'WSR19','WSR20','WSR21','WSR22','WSR23','WSR_PK','WSR_AV','T0','T1','T2','T3'
,'T4','T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15','T16','T17','T18','T19','T
20','T21','T22','T23','T_PK','T_AV','T85','RH85','U85','V85','HT85','T70','RH70','U70
','V70','HT70','T50','RH50','U50','V50','HT50','KI','TT','SLP','SLP_','Precp','Ozone
Day']
eighthr.columns = columns_eighthr
eighthr = eighthr.drop("Date",axis=1)
eighthr.head()
eighthr = eighthr.convert_objects(convert_numeric=True)
eighthr.describe()
eighthr.info()
import missingno as msno
msno.matrix(eighthr)
msno.bar(eighthr)
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
eighthr_norm = ((eighthr-eighthr.min())/(eighthr.max()-eighthr.min()))*20
eighthr_norm.head()
from sklearn.preprocessing import Imputer
i = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)22. eighthr_new = i.fit_transform(eighthr_norm)
eighthr_new
columns_eighthr =
['WSR0','WSR1','WSR2','WSR3','WSR4','WSR5','WSR6','WSR7','WSR8','WSR9','WS
R10','WSR11','WSR12','WSR13','WSR14','WSR15','WSR16','WSR17','WSR18','WSR
19','WSR20','WSR21','WSR22','WSR23','WSR_PK','WSR_AV','T0','T1','T2','T3','T4','
T5','T6','T7','T8','T9','T10','T11','T12','T13','T14','T15','T16','T17','T18','T19','T20','T
21','T22','T23','T_PK','T_AV','T85','RH85','U85','V85','HT85','T70','RH70','U70','V70
','HT70','T50','RH50','U50','V50','HT50','KI','TT','SLP','SLP_','Precp','Ozone Day']
eighthr_df =pd.DataFrame(data = eighthr_new) #eighthr_df holds preprocessed
eighthr dataframe
eighthr_df.columns = columns_eighthr
eighthr_df.head()
eighthr_df.describe()
X = eighthr_df.iloc[:,:-1].values
Y = eighthr_df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,
random_state=9)
fig = plt.figure(figsize = (14,4))
title = fig.suptitle("Peak Temperatures", fontsize=14)
plt.ylabel("Freq")
sns.distplot(eighthr_df['T_PK'], hist=True, kde=True, rug = True)
fig = plt.figure(figsize = (14,4))
title = fig.suptitle("Peak Wind Speeds", fontsize=14)
plt.ylabel("Freq")
sns.distplot(eighthr_df['WSR_PK'], hist=True, kde=True, rug = True)
fig = plt.figure(figsize = (12,12))
subset_attributes =
['WSR_PK','T_PK','T85','T70','RH70','U70','V70','HT70','KI','TT','SLP','SLP_','Precp']43. corr = eighthr_df[subset_attributes].corr()
hm = sns.heatmap(round(corr,2), annot=True,fmt='.2f', linewidths=.05)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4, metric='manhattan')
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred, average = 'weighted'))
print("Recall:", metrics.recall_score(Y_test, Y_pred, average = 'weighted'))
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred, average = 'weighted'))
print("Recall:", metrics.recall_score(Y_test, Y_pred, average = 'weighted'))
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred, average = 'weighted'))
print("Recall:", metrics.recall_score(Y_test, Y_pred, average = 'weighted'))
