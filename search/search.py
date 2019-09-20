import numpy as np
import imghdr
import json
import pickle
import hnswlib
import glob, os, time
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import h5py
import os

name = []
for img_file in os.listdir('../eval'):
    name.append(img_file)
# MAC will generate a redundant file
name = name[1:]
EMBEDDING_SIZE = 512

''' test with a single image
file_test = h5py.File('embedding_test.h5','r')
test = np.array(file_test.get('embedding_test'))
p = hnswlib.Index(space='l2', dim = EMBEDDING_SIZE)  # you can change the sa
p.load_index('index.idx')
q_labels,_=p.knn_query(test,k=1)
'''
p = hnswlib.Index(space='l2', dim = EMBEDDING_SIZE)  # you can change the sa
p.load_index('index.idx')
file = h5py.File('embedding.h5','r')
features = np.array(file.get('embedding'))
# test with the whole feature space
f= open("test.dat","w+")
for i, feature in enumerate(features):
    k = 3
    q_labels,_=p.knn_query(features[i],k=k)
    if '00.jpg' in name[i]:
        f.write(name[i])
        for j in range(k-1):
            f.write(" " + str(j) + " " + name[q_labels[0][j+1]])
        f.write('\n')
f.close()