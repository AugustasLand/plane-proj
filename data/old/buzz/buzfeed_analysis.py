import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix

train = pd.read_csv("planes_features.csv")
labels = pd.read_csv("train.csv")

mask = (labels['class'] == 'surveil')
print(np.sum(mask))
labels.loc[mask, 'class'] = -1 
labels.loc[~mask, 'class'] = 1 

train_check = pd.merge(train, labels, on="adshex")

train = train_check[['duration1', 'duration2', 'duration3', 'duration4', 'duration5',
       'boxes1', 'boxes2', 'boxes3', 'boxes4', 'boxes5', 'speed1', 'speed2',
       'speed3', 'speed4', 'speed5', 'altitude1', 'altitude2', 'altitude3',
       'altitude4', 'altitude5', 'steer1', 'steer2', 'steer3', 'steer4',
       'steer5', 'steer6', 'steer7', 'steer8', 'flights', 'squawk_1',
       'observations']]

# model = IsolationForest(random_state=42)
model = LocalOutlierFactor(n_neighbors=5, contamination=1/5)
# model = GaussianMixture(2)
# model = OneClassSVM(kernel='poly')
prediction = model.fit_predict(train)
acc = np.mean(train_check['class'].values == prediction)
print(confusion_matrix(train_check['class'].to_numpy(dtype=np.int16), prediction))
print(acc)