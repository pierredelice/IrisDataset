#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from params import get_params
from dataset import dataset_model


# In[9]:


params = get_params()
dataset= dataset_model(params)
df = dataset.get_data()


# In[10]:


import matplotlib.pyplot as plt 
import numpy as np 

# select setosa and versicolor 
y = df.iloc[0:100, 4].values 
y = np.where(y == 'Iris-setosa', 0, 1) 

# extract sepal length and petal length 
X = df.iloc[0:100, [0, 2]].values 
# plot data 
plt.scatter(X[:50, 0], X[:50, 1], 
            color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], 
            color='blue', marker='s', label='Versicolor') 
plt.xlabel('Sepal length [cm]') 
plt.ylabel('Petal length [cm]') 
plt.legend(loc='upper left') 
plt.show()


# In[20]:


from mlp import Perceptron

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()


# In[21]:


from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')


# In[22]:


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')


#plt.savefig('images/02_08.png', dpi=300)
plt.show()


# In[ ]:




