
# coding: utf-8

# # Bank Note Authentication using K-Means Clustering
# > Written by Kieran Shand

# ## Data Wrangling
# ### General Properties

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('bank_note.csv')


# In[3]:


df.head()
#Column headers are capitalised


# In[4]:


df.duplicated().value_counts()
#Several dupplicated results


# In[5]:


df.info()
#No null values


# In[6]:


df['Class'].value_counts()
#All observations in the Class variable equal to 1 or 2 


# In[7]:


df.describe()


# In[8]:


g = sb.PairGrid(data = df, vars = ['V1', 'V2', 'V3', 'V4', 'Class'])
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.savefig('image_1.png', dpi = 300);


# ## Modelling using K-Means

# In[58]:


v1 = df['V1']
v4 = df['V4']

variable_stack = np.column_stack((v1, v4))
km_res = KMeans(n_clusters=2).fit(variable_stack)
clusters = km_res.cluster_centers_


# In[59]:


plt.figure(figsize=[12,8])
plt.title('Variable 1 Vs. Variable 4',fontsize=18)
plt.xlabel('Variable 1', fontsize=14)
plt.ylabel('Variable 4',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.scatter(v1, v4, alpha = 0.4, c=df['Class'], cmap='tab10')
plt.savefig('image_2.png', dpi = 300);


# In[60]:


plt.figure(figsize=[12,8])
plt.title('Variable 1 Vs. Variable 4 K-Means Clustering',fontsize=18)
plt.xlabel('Variable 1', fontsize=14)
plt.ylabel('Variable 4',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.scatter(v1, v4, alpha = 0.4, c=df['Class'], cmap='tab10')
plt.scatter(clusters[0], clusters[1], s=500, c='k')
plt.savefig('image_3.png', dpi = 300);

