
# coding: utf-8

# In[12]:


# Loading Data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# In[3]:


# Path of the file to read
file_path = '/home/andy/titanic/train.csv'

titanic_data = pd.read_csv(file_path)
titanic_data['Sex'].replace(['female','male'],[0,1],inplace=True)
# Create target object and call it y
features = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']


# In[4]:


#sorting out X and y, dropping na values from the features set
X = titanic_data[features].dropna()
y = X['Survived'].dropna()
X = X.drop(columns = ['Survived'])


# In[39]:


# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y)


# In[41]:


len(train_X), len(val_X), len(train_y), len(val_y)




# In[33]:


titanic_model = DecisionTreeRegressor(random_state=1)


# In[34]:


#Fit model
titanic_model.fit(train_X,train_y)


# In[16]:


print("Making predictions for the following 5 people:")
print(train_X.head())
print("The predictions are")
print(titanic_model.predict(train_X.head()))


# In[55]:


#illustrates errors using the training data, trying out accuracy_score (a classification method)
predicted_titanic_survival = titanic_model.predict(train_X)
predicted_titanic_survival = predicted_titanic_survival.astype(int)
accuracy_score(train_y, predicted_titanic_survival)


# In[54]:


#validations: 
val_predictions = titanic_model.predict(val_X)
val_predictions = val_predictions.astype(int)
print(accuracy_score(val_y, val_predictions))


# In[62]:


#its fucked up past this point ------
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    titanic_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    titanic_model.fit(train_X, train_y)
    preds_val = titanic_model.predict(val_X)
    preds_val = preds_val.astype(int)
    acc_score = accuracy_score(val_y, preds_val)
    return(acc_score)


# In[63]:


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

