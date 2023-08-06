#!/usr/bin/env python
# coding: utf-8

# In[122]:


get_ipython().system('pip install -U imbalanced-learn')


# In[211]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
from scipy.stats import zscore
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 


# In[212]:


# This dataset contains the Red and White variants of Wine


# In[213]:


df = pd.read_csv("winequality-red.csv")
df


# In[214]:


#Variables in Dataset


# In[215]:


df.head()


# In[216]:


df.tail()


# In[217]:


print(f"The rows and columns in the dataset:{df.shape}")
print(f"\n The column headers in the dataset:{df.columns}")


# # EXPOLATORY DATA ANALYSIS

# In[219]:


df.dtypes


# In[220]:


#Checking the null values


# In[221]:


df.isnull().sum()


# We do not see any missing values in any of the columns of our dataset so we don't have to worry about handling missing data.

# In[222]:


df.info()


# Here none of the coulmns have any object data type value and our label is the only integer value making all the features cloumns as float datatype

# In[223]:


#Description of Dataset:
#Statistical summary of Numerical columns
df.describe()


# Using the describe method I can see the count, mean, standard deviation, minimum, maximum and inter quantile values of our dataset.
# 
# As per my observation:
# 
# 1.There is a big gap between 75% and max values of residual sugar column
# 
# 2.There is a big gap between 75% and max values of free sulfur dioxide column
# 
# 3.There is a huge gap between 75% and max value of total sulfur dioxide column

# In[224]:


df.skew()


# Here we see the skewness infirmation present in our dataset. We will ignore quality since it is our target label in the dataset. Now taking a look at all the feature columns we see that fixed acidity, volatile acidity, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, sulphates and alcohol are all outside the acceptable range of +/-0.5. This skewness indicates outliers being present in our dataset that will need to be treated if required.

# In[225]:


df["quality"].unique()


# # Visualization

# In[226]:


plt.figure(figsize=(11,8))
sns.countplot(x ='quality', data =df)
plt.xlabel('Quality of Red wine')
plt.ylabel('Count of Rows in the dataset')
plt.show()


# In[227]:


index=0
labels = df['quality']
features = df.drop('quality', axis=1)
for col in features.items():
    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y=col[index], data=df, color="orange")
plt.tight_layout()
plt.show()


# With the feature vs label barplot we  are able to see the trend corresponding to the impact each has with respect tp predicting the quality column.
# 
# Observation regarding features compared to the label are:01. fixed acidity vs quality - no fixed pattern 02.volatile acidity vs qualityt- there is a decreasing trend 
#  03. citric acid vs quality - there is an increasing trend 04. residual sugar vs quality - no fixed pattern 05. chlorides vs quality - there is a decreasing trend 06. free sulfur dioxide vs quality - no fixed pattern as it is increasing then decreasing 07. total sulfur dioxide vs quality - no fixed pattern as it is increasing then decreasing 08. density vs quality - no pattern at all 09. pH vs quality - no pattern at all 10. sulphates vs quality - there is an increasing trend 11. alcohol vs quality - there is an increasing trend
# 
# So here we can conclude that to get better quality wine citric acid, sulphates and alcohol columns play a major role.    

# In[228]:


fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15,10))
index =0 
ax = ax.flatten()
for col, value in df.items():
    sns.boxplot(y=col,data=df, ax=ax[index])
    index +=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


# In[229]:


fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(15,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
    sns.distplot(value, ax=ax[index], hist=False, color="g", kde_kws={"shade":True})
    index +=1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()


# The distribution plots show that few of the columns are in normal distribution category showing a proper bell shape curve. However, we do see skewness in most of the feature columns like citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, sulphates and alcohol columns. We are going to ignore the label column since it is a categorical column and will need to fix the imbalance data inside it.
# 
# With respect to the treatment of skewness and outliers I will perform the removal or treatment after I can see the accuracy dependency of the machine learning models.
# 
# 

# In[ ]:





# # Dropping a column

# In[231]:


df= df.drop('free sulfur dioxide',axis=1)
df


# Here free sulfur dioxide and total sulfur dioxide are both indicating towards the same feature of sulfur dioxide therefore I am dropping the free option and keeping just the total option in our dataset.

# # Outlier removal

# In[232]:


df.shape


# In[233]:


# Zscore method

z=np.abs(zscore(df))
thersold=3
np.where(z>3)

df=df[(z<3).all(axis=1)]
df


# Here, I have used the Z score method to get rid of outliers present in our dataset that are not in the acceptable range of +/-0.5 value of skewness.

# In[234]:


df.shape


# In[235]:


# Percentage of Data Loss

data_loss=(1599-1464)/1599*100
data_loss


# After removing the outliers we are checking the data loss percentage by comparing the rows in our original data set and the new data set post removal of the outliers.

# # Splitting the dataset into 2 variables

# In[236]:


X= df.drop('quality', axis=1)
Y= df['quality']


# # Taking care of class imbalance

# In[237]:


Y.value_counts()


# In[238]:


oversample= SMOTE()
X, Y = oversample.fit_resample(X, Y)


# In[239]:


Y.value_counts()


# In[240]:


Y


# # Label Binarization

# In[241]:


Y = Y.apply(lambda y_value:1 if y_value>=7 else 0)
Y


# In[242]:


X


# # Feature Scaling

# In[243]:


scaler = StandardScaler()
X =pd.DataFrame(scaler.fit_transform(X), columns=x.columns)
X


# # Creating the training and testing

# In[244]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)


# # Machine Learning Model for Classification and Evaluation Metrics

# In[253]:


# Classification Model Function

def classify(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=21)
    
    # Training the model
    model.fit(X_train, Y_train)
    
    # Predicting Y_test
    pred = model.predict(X_test)
    
    # Accuracy Score
    acc_score = (accuracy_score(Y_test, pred))*100
    print("Accuracy Acore:", acc_score)
    
    # Classification Report
    class_report = classification_report(Y_test, pred)
    print("\nClassification Report:\n", class_report)
    
    # Cross Validation Score
    cv_score = (cross_val_score(model, X, Y, cv=5).mean())*100
    print("Cross Validation Score:", cv_score)
    
    # Result of accuracy minus cv scores
    result = acc_score - cv_score
    print("\nAccuracy Score - Cross Validation Score is", result)


# In this I have defined a class that will perform the train-test split, training of machine learning model, predicting the label value, getting the accuracy score, generating the classification report, getting the cross validation score and the result of difference between the accuracy score and cross validation score for any machine learning model that calls for this function.

# In[254]:


# Logistic Regression

model=LogisticRegression()
classify(model, X, Y)


# In[255]:


# Support Vector Classifier

model=SVC(C=1.0, kernel='rbf', gamma='auto', random_state=42)
classify(model, X, Y)


# In[256]:


# Decision Tree Classifier

model=DecisionTreeClassifier(random_state=21, max_depth=15)
classify(model, X, Y)


# In[257]:


# Random Forest Classifier

model=RandomForestClassifier(max_depth=15, random_state=111)
classify(model, X, Y)


# # Hyper parameter tuning on the best ML Model

# In[258]:


# Choosing Support Vector Classifier

svc_param = {'kernel': ['poly','sigmod','rbf'],'gama': ['scale','auto'],'shrinking':[True, False],'random_state': [21,42,104],'probability': [True, False], 'decision_function_shape': ['ovo', 'ovr'],'verbose': [True, False]}


# After comparing all the classification models I have selected Support Vector Classifier as my best model and have listed down it's parameters above referring the sklearn webpage.

# In[266]:


GSCV = GridSearchCV(SVC(), svc_param, cv=5)


# Here I am using the Grid Search CV method for hyper parameter tuning my best model.

# In[269]:


GridSearchCV(cv=5, estimator=SVC(),param_grid={'decision_function_shape': ['ovo', 'ovr'],
                         'gamma': ['scale', 'auto'],
                         'kernel': ['poly', 'sigmoid', 'rbf'],
                         'probability': [True, False],
                         'random_state': [21, 42, 104],
                         'shrinking': [True, False], 'verbose': [True, False]})


# In[273]:


Final_Model = SVC(decision_function_shape='ovo', gamma='scale', kernel='rbf', probability=True, random_state=21,
                 shrinking=True, verbose=True)
Classifier = Final_Model.fit(X_train, Y_train)
fmod_pred = Final_Model.predict(X_test)
fmod_acc = (accuracy_score(Y_test, fmod_pred))*100
print("Accuracy score for the Best Model is:", fmod_acc)


# This is the successfull incorporation of the Hyper Parameter Tuning on my Final Model and received the accuracy score for it.
# 
# 

# # AUC ROC Curve

# In[274]:


disp = metrics.plot_roc_curve(Final_Model, X_test, Y_test)
disp.figure_.suptitle("ROC Curve")
plt.show()


# # Saving the Model

# In[276]:


filename = "FinalModel_1.pkl"
joblib.dump(Final_Model, filename)


# In[278]:


import pickle
filename="FinalModel_1.pkl"
pickle.dump(model, open(filename, 'wb'))


# In[ ]:




