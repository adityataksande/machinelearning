import  pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle




data = pd.read_csv("student-mat.csv",sep=";")
#print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#print(data.head())

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(X_train, y_train)

#creating the pickle
with open("studentmodel.pickle","wb") as f:
    pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)
# closing



accuray = linear.score(X_test, y_test)

print(accuray)

print("Coe: \n",linear.coef_)
print("Intercept: \n",linear.intercept_)
predictions = linear.predict(X_test)
for x in range(len(predictions)):
    print(predictions[x], X_test[x], y_test[x])