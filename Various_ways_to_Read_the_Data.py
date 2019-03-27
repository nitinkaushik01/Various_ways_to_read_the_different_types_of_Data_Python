
# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd


# ## 1. Read Data from csv file

# In[7]:

datacsv = pd.read_csv('50_Startups.csv')
datacsv.head()


# ## 2. Load Data using Sklearn Library (Boston Housing Data)

# In[8]:

from sklearn import datasets


# In[9]:

bostondata = datasets.load_boston()


# In[13]:

X = bostondata.data
y = bostondata.target


# In[14]:

X[0]


# In[15]:

y[0]


# ## 3. Read the data from RDBMS Database like MySQL

# In[17]:

#pip install mysql
#pip install mysql-connector-python
db_connect = sql.connect(host='127.0.0.1', database='orderdb', user = 'root', password = 'nose100crib863')


# In[18]:

df = pd.read_sql('SELECT * from order_detail', con=db_connect)
df.head()


# # 4. Simulate the Data

# ## Simulate the Data for Classification Algorithms

# In[19]:

from sklearn.datasets import make_classification


# In[22]:

features, target = make_classification(n_samples =  250, n_features = 8, n_informative = 6, n_redundant = 2, 
                                       n_classes = 4, weights = [.1, .2, .3, .4])


# In[23]:

pd.DataFrame(features).head()


# In[24]:

pd.DataFrame(target).head()


# ## Simulate the Data for Regression Algorithms

# In[25]:

from sklearn.datasets import make_regression


# In[26]:

features, target = make_regression(n_samples =  250, n_features = 4, n_targets = 1)


# In[27]:

pd.DataFrame(features, columns=['Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4']).head()


# In[28]:

pd.DataFrame(target, columns=['Threshold']).head()


# ## Simulate the Data for Clustering Algorithms

# In[34]:

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


# In[44]:

feat, tgt = make_blobs(n_samples =  250, n_features = 3, centers = 4, cluster_std = 0.5, shuffle =  True)


# In[45]:

feat.shape


# In[46]:

plt.scatter(feat[:,0],
           feat[:,1])
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



