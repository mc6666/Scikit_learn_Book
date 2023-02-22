#!/usr/bin/env python
# coding: utf-8

# # 自行開發決策樹
# ### 程式修改自[Implementing Decision Tree From Scratch in Python](https://medium.com/@penggongting/implementing-decision-tree-from-scratch-in-python-c732e7c69aea)

# ## 載入相關套件

# In[8]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import math


# ## 計算熵(entropy)

# In[9]:


# 熵公式
def entropy_func(c, n):
    # return -(c*1.0/n)*math.log(c*1.0/n, 2)
    # gini
    return 1-(c*1.0/n)**2

# 計算同一節點內的熵，只有兩個類別
def entropy_cal(c1, c2):
    if c1== 0 or c2 == 0: 
        return 0
    return entropy_func(c1, c1+c2) + entropy_func(c2, c1+c2)

# 計算同一節點內的熵，多個類別
def entropy_of_one_division(division): 
    s = 0
    n = len(division)
    classes = set(division)
    # 計算同一類別的熵
    for c in classes:   
        n_c = sum(division==c)
        e = n_c*1.0/n * entropy_cal(sum(division==c), sum(division!=c))
        s += e
    return s, n

# 計算分割條件的熵
def get_entropy(y_predict, y_real):
    if len(y_predict) != len(y_real):
        print('They have to be the same length')
        return None
    n = len(y_real)
    # 左節點
    s_true, n_true = entropy_of_one_division(y_real[y_predict]) 
    # 右節點
    s_false, n_false = entropy_of_one_division(y_real[~y_predict])
    # 左、右節點加權總和
    s = n_true*1.0/n * s_true + n_false*1.0/n * s_false 
    return s


# ## 決策樹演算法類別

# In[10]:


class DecisionTreeClassifier(object):
    def __init__(self, max_depth=3):
        self.depth = 0
        self.max_depth = max_depth
    
    # 訓練
    def fit(self, x, y, par_node={}, depth=0):
        if par_node is None: 
            return None
        elif len(y) == 0:
            return None
        elif self.all_same(y):
            return {'val':float(y[0])}
        elif depth >= self.max_depth:
            return None
        else: 
            # 計算資訊增益
            col, cutoff, entropy = self.find_best_split_of_all(x, y) 
            if cutoff is not None:   
                y_left = y[x[:, col] < cutoff]
                y_right = y[x[:, col] >= cutoff]
                par_node = {'col': feature_names[col], 'index_col': col,
                            'cutoff':float(cutoff),
                            'val': float(np.round(np.mean(y)))}
                par_node['left'] = self.fit(x[x[:, col] < cutoff], y_left, {}, depth+1)
                par_node['right'] = self.fit(x[x[:, col] >= cutoff], y_right, {}, depth+1)
                self.depth += 1 
            self.trees = par_node
            return par_node
    
    # 根據所有特徵找到最佳切割條件
    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:    # 找到最佳切割條件
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy
    
    # 根據一個特徵找到最佳切割條件
    def find_best_split(self, col, y):
        min_entropy = 10
        n = len(y)
        for value in set(col):
            y_predict = col < value
            my_entropy = get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff
    
    # 檢查是否節點中所有樣本均屬同一類
    def all_same(self, items):
        return all(x == items[0] for x in items)
                                           
    # 預測
    def predict(self, x):
        tree = self.trees
        results = np.array([0]*len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results
    
    # 預測一筆
    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer is not None and cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val') if cur_layer is not None else None


# ## 載入資料集

# In[11]:


ds = datasets.load_iris()
feature_names = ds.feature_names
X, y = ds.data, ds.target


# ## 資料分割

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# ## 選擇演算法

# ## 模型訓練

# In[13]:


import json

clf = DecisionTreeClassifier()
output = clf.fit(X_train, y_train)
# output
print(json.dumps(output, indent=4))


# ## 模型評分

# In[14]:


# 計算準確率
y_pred = clf.predict(X_test)
print(f'{accuracy_score(y_test, y_pred)*100:.2f}%') 


# In[ ]:




