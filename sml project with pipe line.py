#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor


# In[2]:


df = pd.read_csv("rent.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df['bedroom'] = df['title'].str.findall(r'([\d.]+)').str[0].astype(float)
print(df)


# In[7]:


df.head()


# In[8]:


df.isnull().sum()


# In[9]:


price = df.price.tolist()


# In[10]:


price_int = []
for x in price:
    if type(x) == type(0):
        price_int.append(x)
    elif x[-1] == 'L':
        price_int.append(int(float(x.split()[0])) * 100000)
    elif type(x) == type('p'):
        price_int.append(int(x.replace(',','')))


# In[11]:


price_final=[]
for item in price_int:
    price_final.append(float(item))
print(price_final, end=" ")
print(type(price_final))


# In[12]:


# df = pd.DataFrame(price_int, columns, dtype = float) 
df['rent'] = price_int


# In[13]:


df.head()


# In[14]:


df2=df.drop('price',axis=1)


# In[15]:


sns.boxplot(data = df2,x='rent')


# In[16]:


sns.kdeplot(df2['rent']);


# In[17]:


df2.location =df2.location.apply(lambda x : x.strip())
locationc = df2.groupby('location')['location'].count().sort_values(ascending=False)
locationc


# In[18]:


df2.bedroom.unique()


# In[41]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df2['location']= label_encoder.fit_transform(df2['location'])


# In[42]:


X = df2.drop(['rent', 'title'], axis=1)
y= df2.rent


# In[43]:


x


# In[44]:


y


# In[45]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)


# In[46]:


X_train


# In[47]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[48]:


ct = ColumnTransformer([('one-hot-encoder', OneHotEncoder(drop='first'), [2,3])], remainder='passthrough')


# In[49]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[2,3])
],remainder='passthrough')


# In[50]:


step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])


# In[51]:


pipe.fit(X_train,y_train)


# In[54]:


y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[55]:


#Ridge


# In[56]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[2,3])
],remainder='passthrough')


# In[58]:


step2 = Ridge(alpha=10)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])


# In[59]:


pipe.fit(X_train,y_train)


# In[60]:


y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[61]:


#laso


# In[62]:


step1 = ColumnTransformer(transformers=[
    ('col_tnf',OneHotEncoder(sparse=False,drop='first'),[2,3])
],remainder='passthrough')

step2 = Lasso(alpha=0.001)

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])

pipe.fit(X_train,y_train)

y_pred = pipe.predict(X_test)

print('R2 score',r2_score(y_test,y_pred))
print('MAE',mean_absolute_error(y_test,y_pred))


# In[ ]:




