
# coding: utf-8

# In[3]:


import numpy as np
import imghdr
import json
import pickle
import hnswlib
import glob, os, time
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import h5py


# In[50]:


 file = h5py.File('embedding.h5','r')
 features = np.array(file.get('embedding'))

for x in xrange(2, 30):
	file = h5py.File('embedding_{}.h5'.format(x*50))
	feature = np.array(file.get('embedding'))
	features = np.concatenate((features,feature))

file = h5py.File('embedding_1491.h5','r')
feature = np.array(file.get('embedding'))
features = np.concatenate((features,feature))


# In[51]:


num_elements = len(features)
labels_index = np.arange(num_elements)


# In[52]:


# Declaring index
EMBEDDING_SIZE = 512
p = hnswlib.Index(space = 'l2', dim = EMBEDDING_SIZE) # possible options are l2, cosine or ip


# In[53]:


# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 100, M = 16)


# In[54]:


# Element insertion (can be called several times):
int_labels = p.add_items(features, labels_index)

# Controlling the recall by setting ef:
p.set_ef(100) # ef should always be > k
p.save_index('index.idx')

