import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection
import pandas as pd



df = pd.read_excel('data_sets/titanic.xls')
df.drop(['body', 'name'], 1 , inplace=True)
df.fillna(0, inplace=True)

# This function tries to turn data into ints. If it's a string converts to int
def handle_non_num_data(df):
    columns = df.columns.values

    for column in columns:
        try:
            df[column] = df[column].astype(np.float64)
        except:
            print("skipping.")
        text_digit_value = {}
        def convert_to_int(val):
            return text_digit_value[val]

        if df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique in unique_elements:
                    if unique not in text_digit_value:
                        text_digit_value[unique] = x
                        x += 1
            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_num_data(df)

#print(df.head())

X = np.array(df.drop(['survived'], 1))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)



# testing for survival
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i])
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
