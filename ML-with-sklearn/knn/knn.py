import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from  sklearn import linear_model ,preprocessing

df = pd.read_csv("car.data")
print(df.head())

# getting dummies done
