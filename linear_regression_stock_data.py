import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

# Uses Quandl to get data for google stock prices and changes the columns to relevant data
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

#Fills empty cells with number to create outliers
df.fillna(-99999, inplace=True)


# Makes the label and adds a column
forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df)))
print("Days predicted out: ",forecast_out)

# Moves the column up to get a future price that we can try to predict
df['label'] = df[forecast_col].shift(-forecast_out)


# Takes Data from Quandl and puts it inside array (which is how the modules like sklearn look at info)
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

# Removes N/A data or nan data
df.dropna(inplace=True)
# Makes the Numbers smaller so it can be processed by the computer faster
y = np.array(df['label'])

# Splits up the data into two groups
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Starts the linear regression model
clf = LinearRegression()

# Fit == Train
clf.fit(X_train, y_train)

#Look at the results of testing
accuracy = clf.score(X_test, y_test)
print("accuracy: ", accuracy)

# Using the model to predict future days 
forecast_set = clf.predict(X_lately)

# Printing the results
print(df.tail())
print(forecast_set)




# Graphing the results

# style option for the graph
style.use('ggplot')

# Creates a column of nans
df['Forecast'] = np.nan

# This whole part just makes the days for the graph
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Data')
plt.ylabel('Price')
plt.show()

