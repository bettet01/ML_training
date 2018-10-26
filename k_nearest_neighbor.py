import numpy as np
from sklearn import preprocessing, neighbors, model_selection
import pandas as pd

df = pd.read_csv('data_sets/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('Accuracy:',str(round(accuracy*100, 2)) + '%')

example_measures = np.array([[4,2,3,6,3,5,4,2,4],[4,2,1,1,1,5,6,2,5]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction)

