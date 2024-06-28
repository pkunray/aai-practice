# Gini Impurity tells us what is the probability of misclassifying an observation. 
# Note that the lower the Gini the better the split. 
# In other words the lower the likelihood of misclassification.
# 1 – (p₁)² – (p₂)²
# where p₁ is the probability of class 1 and p₂ is the probability of class

# Note: Random Forest vs Decision Tree

from sklearn import datasets,tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = datasets.load_wine()

data_train, data_test, target_train, target_test = train_test_split(dataset.data, dataset.target, train_size = 0.8,random_state=10) 

model = tree.DecisionTreeClassifier(criterion='gini', random_state=0, max_depth=5)
model.fit(data_train, target_train)
result = model.predict(data_test)

score = accuracy_score(target_test, result)
print(model.get_depth())
print(score)

# plt.figure(figsize=(15,10))
# tree.plot_tree(model, 
#                feature_names=dataset.feature_names,  
#                class_names=dataset.target_names, 
#                filled=True, 
#                rounded=True)
# plt.show()

# Instead of reading data from sklearn, I want to read data from a CSV file online using pandas.
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = pd.read_csv(url, header=None)
