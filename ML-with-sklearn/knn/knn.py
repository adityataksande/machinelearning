import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from  sklearn import linear_model ,preprocessing

df = pd.read_csv("car.data")
print(df.head())

# getting dummies done

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(df["buying"]))
maint = le.fit_transform(list(df["maint"]))
door = le.fit_transform(list(df["door"]))
persons = le.fit_transform(list(df["persons"]))
lug_boot = le.fit_transform(list(df["lug_boot"]))
safety = le.fit_transform(list(df["safety"]))
cls = le.fit_transform(list(df["class"]))
# print(buying)


predict = "class"


X = list(zip(buying, maint, door, persons, lug_boot, safety))

y = list(cls)



X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# print(X_train,y_test)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(X_train,y_train)

acc= model.score(X_test,y_test)
print(acc)

predicted = model.predict(X_test)
names = ["unacc", "acc", "good", "verygood"]

for x in range(len(X_test)):
    print("Predicted : ",predicted[x], "Data: ", X_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([X_test[x]], 9, True)
    print(n)

