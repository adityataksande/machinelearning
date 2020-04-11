import matplotlib.pyplot as plt
from sklearn import datasets,svm

digits  = datasets.load_digits()

clf = svm.SVC(gamma=0.0001,C=100)

print(len(digits.data))

X,y = digits.data[:-1], digits.target[:-1]

clf.fit(X,y)

print("Prediction :" , clf.predict([digits.data[-6]]))

plt.imshow(digits.images[-6],cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()